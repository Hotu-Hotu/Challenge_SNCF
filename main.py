import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


start = time.time()
TARGET = "p0q0"
VALID_DAYS = 14
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


#------------------------------
#------------ chargement donnees ------------
# ------------------------------
x_train = pd.read_csv("x_train_final.csv")
x_test = pd.read_csv("x_test_final.csv")
y_train = pd.read_csv("y_train_final.csv")

x_train = x_train.drop(columns=[c for c in x_train.columns if str(c).startswith("Unnamed")])
x_test = x_test.drop(columns=[c for c in x_test.columns if str(c).startswith("Unnamed")])
y_train = y_train.drop(columns=[c for c in y_train.columns if str(c).startswith("Unnamed")])[TARGET]

x_train["test"] = 0
x_test["test"] = 1
df = pd.concat([x_train, x_test], ignore_index=True)


#------------------------------
#------------ feature engineering ------------
# ------------------------------
date = pd.to_datetime(df["date"])
df["day_of_week"] = date.dt.dayofweek
df["day_of_month"] = date.dt.day

lag_cols = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
df["sum_pq"] = df[lag_cols].sum(axis=1)
df["mean_pq"] = df[lag_cols].mean(axis=1)
df["std_pq"] = df[lag_cols].std(axis=1)

df_train = df[df["test"] == 0].drop(columns="test").reset_index(drop=True)
df_test = df[df["test"] == 1].drop(columns="test").reset_index(drop=True)




# --------------------------------
#------------ Graph --------------
#----------------------------------
G = nx.DiGraph()
for _, group_train in df_train.sort_values(["date", "train", "arret"]).groupby(["date", "train"]):
    gares = group_train["gare"].values
    stops = group_train["arret"].values
    delays = group_train["p2q0"].values
    for i in range(len(gares) - 1):
        if stops[i + 1] != stops[i] + 1:
            continue
        edge = (gares[i], gares[i + 1])
        old = G.edges[edge] if G.has_edge(*edge) else {"delay": 0, "count": 0}
        G.add_edge(*edge, delay=old["delay"] + delays[i + 1], count=old["count"] + 1)

flow_rows = []
for gare in sorted(set(df_train["gare"]).union(set(df_test["gare"]))):
    values = []
    if gare in G and len(nx.ancestors(G, gare)) > 0:
        for edge in nx.bfs_edges(G, source=gare, depth_limit=6, reverse=True):
            reverse_edge = edge[::-1]
            if reverse_edge in G.edges:
                values.append([G.edges[reverse_edge]["delay"], G.edges[reverse_edge]["count"]])
    values = np.array(values, dtype="float64") if values else np.zeros((0, 2))
    count_sum = values[:, 1].sum() if len(values) else 0
    flow_rows.append({"gare": gare, "flowavg": 0 if count_sum == 0 else (values[:, 0] / count_sum).sum()})

df_train = df_train.merge(pd.DataFrame(flow_rows), on="gare", how="left")
df_test = df_test.merge(pd.DataFrame(flow_rows), on="gare", how="left")

hist = df_train.copy()
hist[TARGET] = y_train.values
global_mean = float(hist[TARGET].mean())
global_std = float(hist[TARGET].std())


# ------------------------------
# --------------- Target encoding --------------
# ----------------------------

stat_tables = [
    hist.groupby("gare").agg(
        gare_target_mean=(TARGET, "mean"),
        gare_target_median=(TARGET, "median"),
        gare_target_std=(TARGET, "std"),
        gare_count=(TARGET, "size"),
        gare_arret_mean=("arret", "mean"),
        gare_p2q0_mean=("p2q0", "mean"),
        gare_p0q2_mean=("p0q2", "mean"),
    ).reset_index(),
    hist.groupby(["gare", "arret"]).agg(
        gare_arret_target_mean=(TARGET, "mean"),
        gare_arret_target_median=(TARGET, "median"),
        gare_arret_count=(TARGET, "size"),
    ).reset_index(),
    hist.groupby(["gare", "day_of_week"]).agg(
        gare_dow_target_mean=(TARGET, "mean"),
        gare_dow_count=(TARGET, "size"),
    ).reset_index(),
]

for data_name in ["df_train", "df_test"]:
    data = locals()[data_name]
    for table in stat_tables:
        merge_cols = [c for c in ["gare", "arret", "day_of_week"] if c in table.columns]
        data = data.merge(table, on=merge_cols, how="left")
    data["gare_target_std"] = data["gare_target_std"].fillna(global_std)
    for col in ["gare_count", "gare_arret_count", "gare_dow_count"]:
        data[col] = data[col].fillna(0)
    for col in [
        "gare_target_mean",
        "gare_target_median",
        "gare_arret_target_mean",
        "gare_arret_target_median",
        "gare_dow_target_mean",
    ]:
        data[col] = data[col].fillna(global_mean)
    locals()[data_name] = data





#------------------------------
#------------ Split ------------
# ------------------------------
valid_dates = set(sorted(df_train["date"].unique())[-VALID_DAYS:])
train_mask = ~df_train["date"].isin(valid_dates)
valid_mask = df_train["date"].isin(valid_dates)

x_fit = df_train.loc[train_mask].reset_index(drop=True)
y_fit = y_train.loc[train_mask].reset_index(drop=True)
x_valid = df_train.loc[valid_mask].reset_index(drop=True)
y_valid = y_train.loc[valid_mask].reset_index(drop=True)






#------------------------------
#------------ models ------------
# ------------------------------
categorical_features = ["gare"]
numeric_features = [
    "arret", 
    "p2q0", "p3q0", "p4q0", 
    "p0q2", "p0q3", "p0q4",
    "day_of_week",
    "sum_pq", "mean_pq", "std_pq",
    "gare_target_mean", "gare_target_median", "gare_target_std", "gare_count",
    "gare_arret_mean", "gare_p2q0_mean", "gare_p0q2_mean",
    "gare_arret_target_mean", "gare_arret_target_median", "gare_arret_count",
    "gare_dow_target_mean", "gare_dow_count", "flowavg",
]
features = categorical_features + numeric_features

lgbm_params = [
    dict(objective="regression_l1", 
         n_estimators=900, 
         learning_rate=0.03, 
         num_leaves=63, 
         min_child_samples=40,
         subsample=0.9, 
         colsample_bytree=0.9, 
         reg_lambda=0.2, 
         n_jobs=-1, 
         random_state=SEED, 
         verbose=-1),

    dict(objective="regression_l1", 
         n_estimators=1200, 
         learning_rate=0.02, 
         num_leaves=31, 
         min_child_samples=70,
         subsample=0.85, 
         colsample_bytree=0.8, 
         reg_lambda=0.8, 
         n_jobs=-1, 
         random_state=SEED + 1, 
         verbose=-1),

]
rf_params = [
    dict(n_estimators=100, 
         max_depth=18, 
         max_features="sqrt", 
         n_jobs=1, 
         random_state=SEED),

    dict(n_estimators=100, 
         max_depth=12,  
         max_features=0.7, 
         n_jobs=1, 
         random_state=SEED + 1),
]







#------------------------------
#------------ Train ------------
# ------------------------------

#Encodage catégorielle pour lgbm
station_map = {
    gare: idx + 1
    for idx, gare in enumerate(sorted(
        set(df_train["gare"].astype(str)).union(df_test["gare"].astype(str))
    ))
}
for data in [x_fit, x_valid, df_train, df_test]:
    data["gare"] = data["gare"].astype("category")

# Aligner les catégories valid/test sur train
x_valid["gare"] = x_valid["gare"].cat.set_categories(x_fit["gare"].cat.categories)
df_test["gare"]  = df_test["gare"].cat.set_categories(df_train["gare"].cat.categories)








valid_preds = {}
test_preds  = {}

for i, params in enumerate(lgbm_params, start=1):
    name = f"lgbm_{i}"
    m = LGBMRegressor(**params).fit(x_fit[features], y_fit, categorical_feature=categorical_features)
    valid_preds[name] = m.predict(x_valid[features])

    m = LGBMRegressor(**params).fit(df_train[features], y_train, categorical_feature=categorical_features)
    test_preds[name] = m.predict(df_test[features])

    print(f"{name} MAE : {mean_absolute_error(y_valid, valid_preds[name].round()):.6f}")

for data in [x_fit, x_valid, df_train, df_test]:
    data["gare_code"] = data["gare"].astype(str).map(station_map).fillna(0).astype("int32")

rf_features = numeric_features + ["gare_code"]
for i, params in enumerate(rf_params, start=1):
    name = f"rf_{i}"
    m = RandomForestRegressor(**params).fit(x_fit[rf_features], y_fit)
    valid_preds[name] = m.predict(x_valid[rf_features])

    m = RandomForestRegressor(**params).fit(df_train[rf_features], y_train)
    test_preds[name] = m.predict(df_test[rf_features])

    print(f"{name} MAE : {mean_absolute_error(y_valid, valid_preds[name].round()):.6f}")












#Aggrégation avec nn
X_meta_fit   = np.column_stack(list(valid_preds.values())).astype("float32")
X_meta_train = np.column_stack(list(test_preds.values())).astype("float32")

meta_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X_meta_fit.shape[1],)),
    tf.keras.layers.Dense(8,  activation="relu"),
    tf.keras.layers.Dense(1),
])
meta_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mae")
meta_model.fit(
    X_meta_fit, y_valid.to_numpy(dtype="float32"),
    epochs=100,
    batch_size=512,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)],
    verbose=0,
)

pred_valid_meta = meta_model.predict(X_meta_fit, verbose=0).ravel()
print(f"NN stacking MAE : {mean_absolute_error(y_valid, pred_valid_meta.round()):.6f}")



#------------------------------
#------------ Pred -------------
# ------------------------------

test_pred = meta_model.predict(X_meta_train, verbose=0).ravel().round().astype(int)

pd.DataFrame({TARGET: test_pred}).to_csv("predictions.csv", index=True)
print(f"Predictions générées : {len(test_pred)} lignes")
print(f"Temps d'exécution : {time.time() - start:.2f} secondes")