import joblib
model = joblib.load(
    "models/random_forest.pkl"
)
sample = [[
    0,
    8,
    90,
    200,
    1,
    95
]]
prediction = model.predict(sample)

print("Prediction:", prediction)