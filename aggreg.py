from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


DATA_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
LAG_COLUMNS = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
P_LAG_COLUMNS = ["p2q0", "p3q0", "p4q0"]
Q_LAG_COLUMNS = ["p0q2", "p0q3", "p0q4"]


def load_data():
    x_train = pd.read_csv(DATA_DIR / "data/x_train_final.csv", index_col=0)
    y_train = pd.read_csv(DATA_DIR / "data/y_train_final.csv", index_col=0)
    x_test = pd.read_csv(DATA_DIR / "data/x_test_final.csv", index_col=0)

    if "Unnamed: 0" in x_train.columns:
        x_train = x_train.drop(columns="Unnamed: 0")

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["test"] = 0
    x_test["test"] = 1

    full_x = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    return full_x, y_train["p0q0"], len(x_train), x_test.index


def add_group_mean_features(df, group_columns, value_columns, prefix):
    grouped = df.groupby(group_columns, dropna=False)[value_columns]
    means = grouped.transform("mean")
    means.columns = [f"{prefix}_{column}_mean" for column in value_columns]
    return pd.concat([df, means], axis=1)


def add_basic_features(df):
    df = df.copy()
    parsed_date = pd.to_datetime(df["date"])

    df["dayofweek"] = parsed_date.dt.dayofweek

    df["p_lag_mean"] = df[P_LAG_COLUMNS].mean(axis=1)
    df["q_lag_mean"] = df[Q_LAG_COLUMNS].mean(axis=1)
    df["p_q_lag_diff"] = df["p_lag_mean"] - df["q_lag_mean"]
    df["p_lag_trend"] = df["p2q0"] - df["p4q0"]
    df["q_lag_trend"] = df["p0q2"] - df["p0q4"]

    df = add_group_mean_features(df, "gare", ["p_lag_mean", "q_lag_mean"], "gare")
    df = add_group_mean_features(df, "arret", ["p_lag_mean", "q_lag_mean"], "arret")
    df = add_group_mean_features(df, ["gare", "arret"], ["p_lag_mean"], "gare_arret")
    df = add_group_mean_features(df, ["date", "gare"], ["p_lag_mean"], "date_gare")
    return df


def build_flow_graph(df):
    sub_graphs = {}
    graph_data = df[["date", "train", "gare", "arret", "p2q0", "p0q2"]].copy()

    for day, group_day in graph_data.groupby("date"):
        sub_graph = nx.DiGraph()
        for _, group_train in group_day.groupby("train"):
            group_train = group_train.sort_values("arret")
            gares = group_train["gare"].to_numpy()
            stops = group_train["arret"].to_numpy()
            delays = group_train["p2q0"].to_numpy()
            delays_s2 = group_train["p0q2"].to_numpy()

            edges = [
                (gares[i], gares[i + 1])
                for i in range(len(gares) - 1)
                if stops[i + 1] == stops[i] + 1
            ]
            sub_graph.add_edges_from(edges)

            for i, edge in enumerate(edges):
                edge_delay = delays[i + 1]
                edge_delay_s2 = delays_s2[i + 1]
                if sub_graph.edges[edge]:
                    sub_graph.edges[edge]["delay"] += edge_delay
                    sub_graph.edges[edge]["count"] += 1
                    sub_graph.edges[edge]["delay_s2"] += edge_delay_s2
                else:
                    nx.set_edge_attributes(
                        sub_graph,
                        {edge: {"delay": edge_delay, "count": 1, "delay_s2": edge_delay_s2}},
                    )

        sub_graphs[str(pd.Timestamp(day).date())] = sub_graph.copy()

    graph = nx.DiGraph()
    for sub_graph in sub_graphs.values():
        graph = nx.compose(graph, sub_graph)

    edge_data = {}
    for edge in graph.edges:
        edge_data[edge] = {"delay": 0.0, "count": 0, "delay_s2": 0.0}
        for sub_graph in sub_graphs.values():
            if edge in sub_graph.edges:
                edge_data[edge]["delay"] += sub_graph.edges[edge]["delay"]
                edge_data[edge]["count"] += sub_graph.edges[edge]["count"]
                edge_data[edge]["delay_s2"] += sub_graph.edges[edge]["delay_s2"]

    nx.set_edge_attributes(graph, edge_data)
    return graph


def all_flow(graph, target, depth=6):
    if target not in graph:
        return 0.0
    if len(nx.ancestors(graph, target)) == 0:
        return 0.0

    edge_values = [
        [graph.edges[edge[::-1]]["delay"], graph.edges[edge[::-1]]["count"]]
        for edge in nx.bfs_edges(graph, source=target, depth_limit=depth, reverse=True)
    ]
    if not edge_values:
        return 0.0

    edge_values = np.array(edge_values, dtype=float)
    total_count = edge_values[:, 1].sum()
    if total_count == 0:
        return 0.0
    return float((edge_values[:, 0] / total_count).sum())


def add_graph_features(df):
    graph = build_flow_graph(df)
    gares = pd.Series(df["gare"].unique())
    flow_features = pd.DataFrame(
        {
            f"flowavg{depth}": [all_flow(graph, gare, depth) for gare in gares]
            for depth in range(1, 9)
        }
    )
    flow_features["gare"] = gares.values
    return df.merge(flow_features, on="gare", how="left")


def build_preprocessor(x_train):
    categorical_columns = x_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = x_train.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
        ]
    )


def build_base_models():
    return [
        (
            "rf_depth16_leaf5",
            RandomForestRegressor(
                n_estimators=70,
                max_depth=16,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "rf_depth22_leaf3",
            RandomForestRegressor(
                n_estimators=70,
                max_depth=22,
                min_samples_leaf=3,
                max_features=0.7,
                random_state=RANDOM_STATE + 1,
                n_jobs=-1,
            ),
        ),
        (
            "lgbm_leaf31",
            LGBMRegressor(
                n_estimators=600,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="regression_l1",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        (
            "lgbm_leaf63",
            LGBMRegressor(
                n_estimators=450,
                learning_rate=0.045,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                objective="regression_l1",
                random_state=RANDOM_STATE + 1,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
    ]


def build_meta_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(12, 6),
                    activation="tanh",
                    solver="lbfgs",
                    alpha=0.01,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_stack_features(prediction_columns):
    prediction_columns = np.asarray(prediction_columns).T
    return np.column_stack(
        [
            prediction_columns,
            prediction_columns.mean(axis=1),
            np.median(prediction_columns, axis=1),
            prediction_columns.min(axis=1),
            prediction_columns.max(axis=1),
            prediction_columns.std(axis=1),
        ]
    )


def fit_stacked_model(x_train, y, x_test):
    x_base, x_meta, y_base, y_meta = train_test_split(
        x_train,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(x_train)
    x_base_encoded = preprocessor.fit_transform(x_base)
    x_meta_encoded = preprocessor.transform(x_meta)
    x_test_encoded = preprocessor.transform(x_test)

    meta_train_predictions = []
    test_predictions = []
    base_scores = []

    for name, model in build_base_models():
        print(f"Fitting base model: {name}", flush=True)
        model.fit(x_base_encoded, y_base)

        meta_pred = model.predict(x_meta_encoded)
        test_pred = model.predict(x_test_encoded)
        meta_train_predictions.append(meta_pred)
        test_predictions.append(test_pred)

        model_mae = mean_absolute_error(y_meta, meta_pred)
        base_scores.append((name, model_mae))
        print(f"{name} meta MAE: {model_mae:.4f}", flush=True)

    x_meta_stack = build_stack_features(meta_train_predictions)
    x_test_stack = build_stack_features(test_predictions)

    meta_model = build_meta_model()
    print("Fitting neural stacker", flush=True)
    y_mean = y_meta.mean()
    y_std = y_meta.std()
    meta_model.fit(x_meta_stack, (y_meta - y_mean) / y_std)

    meta_predictions = meta_model.predict(x_meta_stack) * y_std + y_mean
    stacked_mae = mean_absolute_error(y_meta, meta_predictions)
    print(f"Stacked validation MAE: {stacked_mae:.4f}", flush=True)

    best_base_name, best_base_mae = min(base_scores, key=lambda item: item[1])
    best_base_index = [name for name, _ in base_scores].index(best_base_name)
    best_base_meta_pred = meta_train_predictions[best_base_index]
    best_base_test_pred = test_predictions[best_base_index]

    neural_test_predictions = meta_model.predict(x_test_stack) * y_std + y_mean
    blend_candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        blend_pred = (1 - alpha) * best_base_meta_pred + alpha * meta_predictions
        blend_mae = mean_absolute_error(y_meta, blend_pred)
        blend_candidates.append((blend_mae, alpha))

    blend_mae, best_alpha = min(blend_candidates, key=lambda item: item[0])
    final_predictions = (1 - best_alpha) * best_base_test_pred + best_alpha * neural_test_predictions
    print(
        f"Best neural blend alpha: {best_alpha:.2f}, validation MAE: {blend_mae:.4f}",
        flush=True,
    )

    if stacked_mae > best_base_mae:
        print(
            f"Warning: neural stacker is worse than {best_base_name} "
            f"({stacked_mae:.4f} vs {best_base_mae:.4f}).",
            flush=True,
        )

    return final_predictions, blend_mae


def main():
    full_x, y, train_size, test_index = load_data()
    print("Data loaded", flush=True)

    # Feature engineering sur le tableau complet train + test.
    full_x = add_basic_features(full_x)
    print("Basic features added", flush=True)
    full_x = add_graph_features(full_x)
    print("Graph features added", flush=True)

    train_mask = full_x["test"].eq(0)
    test_mask = full_x["test"].eq(1)

    x_train = full_x.loc[train_mask].drop(columns="test")
    x_test = full_x.loc[test_mask].drop(columns="test")

    test_predictions, mae = fit_stacked_model(x_train, y, x_test)

    submission = pd.DataFrame({"p0q0": test_predictions}, index=test_index)
    output_path = DATA_DIR / "submission_stacking_rf_lgbm_nn.csv"
    submission.to_csv(output_path)
    print(f"Saved: {output_path}")
    print(f"Train rows: {train_size}, test rows: {len(x_test)}")


if __name__ == "__main__":
    main()
