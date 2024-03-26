import xgboost as xgb
import polars as pl
from sklearn.metrics import accuracy_score

df = pl.read_parquet("parquets/new2.parquet")

# X = df.drop(["matchId", "winning_team"])
X = df.select(["totalGold100", "totalGold200"])
y = df["winning_team"]

# X = X.with_columns(pl.col('ver').cast(pl.Categorical))
# categorical_columns = ["ver", ""]
# X = X.drop(categorical_columns)

dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X, label=y)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1  
}

model = xgb.train(params, dtrain)

predictions = model.predict(dtest)
predictions[predictions < 150.0] = 100.0
predictions[predictions >= 150.0] = 200.0

print(f"Accuracy (Overall) = {accuracy_score(y, predictions):%}")
print(f"Accuracy (10-20 Min) = {accuracy_score(y.filter((df['timestamp'] >= 600) & (df['timestamp'] <= 1200)), predictions[(df['timestamp'] >= 600) & (df['timestamp'] <= 1200)]):%}")
print(f"Accuracy (Game End) = {accuracy_score(y.filter(df['event_type'] == 'GAME_END'), predictions[df['event_type'] == 'GAME_END']):%}")