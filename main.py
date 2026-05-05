from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


data_dir = Path(__file__).resolve().parent
random_state = 42

x_train = pd.read_csv(data_dir / "x_train_final.csv", index_col=0)
y_train = pd.read_csv(data_dir / "y_train_final.csv", index_col=0).squeeze("columns")
x_test = pd.read_csv(data_dir / "x_test_final.csv", index_col=0)

x_train = x_train.drop(columns=["Unnamed: 0"], errors="ignore")
x_test = x_test.drop(columns=["Unnamed: 0"], errors="ignore")

x_train["test"] = 0
x_test["test"] = 1

data = pd.concat([x_train, x_test], axis=0)

date = pd.to_datetime(data["date"])
data["dayofweek"] = date.dt.dayofweek
data["day"] = date.dt.day
data["month"] = date.dt.month

data = data.drop(columns=["train", "date"])

x_train = data[data["test"] == 0].drop(columns=["test"])
x_test = data[data["test"] == 1].drop(columns=["test"])

valid_dates = pd.to_datetime(pd.read_csv(data_dir / "x_train_final.csv", index_col=0)["date"])
valid_mask = valid_dates >= "2023-10-16"

categorical_cols = ["gare"]
numeric_cols = [
    "arret",
    "p2q0",
    "p3q0",
    "p4q0",
    "p0q2",
    "p0q3",
    "p0q4",
    "dayofweek",
    "day",
    "month",
]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_cols,
        ),
        ("num", "passthrough", numeric_cols),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=random_state,
    n_jobs=1,
)

pipeline = Pipeline([("prep", preprocess), ("rf", model)])

pipeline.fit(x_train.loc[~valid_mask], y_train.loc[~valid_mask])
valid_pred = pipeline.predict(x_train.loc[valid_mask])
mae = mean_absolute_error(y_train.loc[valid_mask], valid_pred)
print(f"Temporal validation MAE: {mae:.5f}")

pipeline.fit(x_train, y_train)
test_pred = pipeline.predict(x_test)

submission = pd.DataFrame({"p0q0": test_pred}, index=x_test.index)
submission.to_csv(data_dir / "submission_random_forest.csv")
print("Saved submission_random_forest.csv")
