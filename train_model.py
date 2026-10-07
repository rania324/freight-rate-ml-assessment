import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. File paths
# =========================

train_path = "data/train-test.csv"
validation_path = "data/validation.csv"
december_path = "december-chart-inputs.csv"


# =========================
# 2. Load data
# =========================

df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)
december_df = pd.read_csv(december_path)

print("Training data:", df.shape)
print("Validation data:", validation_df.shape)


# =========================
# 3. Prepare data
# =========================

data = df.copy()

data["date"] = pd.to_datetime(data["date"])

# Replace invalid negative weights
data.loc[data["weight"] < 0, "weight"] = None

# Fill missing values
data["weight"] = data["weight"].fillna(data["weight"].median())
data["market_index"] = data["market_index"].fillna(
    data["market_index"].median()
)


# =========================
# 4. Feature engineering
# =========================

def prepare_features(data):
    data = data.copy()

    data["date"] = pd.to_datetime(data["date"])

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek

    data["route"] = data["pickup"] + "_" + data["delivery"]

    return data


data = prepare_features(data)


# =========================
# 5. Time-based split
# =========================

train_data = data[data["date"] < "2025-10-01"].copy()
val_data = data[data["date"] >= "2025-10-01"].copy()

print("Training split:", train_data.shape)
print("Validation split:", val_data.shape)


# =========================
# 6. Features
# =========================

features = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "day",
    "day_of_week",
    "route"
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

target = "posted_rate"

X_train = train_data[features]
y_train = train_data[target]

X_val = val_data[features]
y_val = val_data[target]


# =========================
# 7. Train validation model
# =========================

model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100
)


# =========================
# 8. Validation results
# =========================

val_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, val_pred)
rmse = mean_squared_error(y_val, val_pred) ** 0.5
r2 = r2_score(y_val, val_pred)

print("\nValidation Results")
print("------------------")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.3f}")

best_iteration = model.get_best_iteration()
print("Best iteration:", best_iteration)


# =========================
# 9. Train final model
# =========================

final_model = CatBoostRegressor(
    iterations=208,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=100
)

X_full = data[features]
y_full = data[target]

final_model.fit(
    X_full,
    y_full,
    cat_features=categorical_features
)


# =========================
# 10. Validation predictions
# =========================

validation_data = validation_df.copy()

validation_data["date"] = pd.to_datetime(
    validation_data["date"]
)

validation_data.loc[
    validation_data["weight"] < 0, "weight"
] = None

validation_data["weight"] = validation_data["weight"].fillna(
    data["weight"].median()
)

validation_data["market_index"] = validation_data[
    "market_index"
].fillna(data["market_index"].median())

validation_data = prepare_features(validation_data)

X_validation = validation_data[features]

validation_pred = final_model.predict(X_validation)

validation_predictions = pd.DataFrame({
    "load_id": validation_df["load_id"],
    "predicted_rate": validation_pred
})

validation_predictions.to_csv(
    "validation_predictions.csv",
    index=False
)

print("\nValidation predictions saved.")


# =========================
# 11. December predictions
# =========================

december_data = december_df.copy()

december_data["date"] = pd.to_datetime(
    december_data["date"]
)

# Historical rows for the same route and equipment
matching_rows = data[
    (data["pickup"] == "Lexington") &
    (data["delivery"] == "Fort Wayne") &
    (data["equipment"] == "Dry Van")
]

# Fill features that are not available
# in the December input file
for col in [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal"
]:
    december_data[col] = matching_rows[col].median()


december_data["year"] = december_data["date"].dt.year
december_data["month"] = december_data["date"].dt.month
december_data["day"] = december_data["date"].dt.day
december_data["day_of_week"] = (
    december_data["date"].dt.dayofweek
)

december_data["route"] = (
    december_data["pickup"]
    + "_"
    + december_data["delivery"]
)

X_december = december_data[features]

december_pred = final_model.predict(X_december)

december_output = december_df.copy()

december_output["predicted_rate"] = december_pred

december_output.to_csv(
    "december-chart-inputs.csv",
    index=False
)

print("December predictions saved.")

print("\nDone!")