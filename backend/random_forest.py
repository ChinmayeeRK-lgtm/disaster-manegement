import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import joblib

data = pd.read_csv(
    "datasets/risk_data.csv"
)

encoder = LabelEncoder()

data["disaster_type"] = encoder.fit_transform(
    data["disaster_type"]
)

data["rescue_needed"] = encoder.fit_transform(
    data["rescue_needed"]
)

data["severity"] = encoder.fit_transform(
    data["severity"]
)

X = data[
    [
        "disaster_type",
        "water_level",
        "building_damage_percent",
        "people_affected",
        "rescue_needed",
        "area_risk_score"
    ]
]

y = data["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

joblib.dump(
    model,
    "models/random_forest.pkl"
)

print("Random Forest Model Saved")