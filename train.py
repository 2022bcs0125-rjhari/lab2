import json, joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- INFO ----------------
NAME = "R J Hari"
ROLL = "2022BCS0125"

# ---------------- PATHS ----------------
DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "model.pkl"
RESULTS_PATH = OUTPUT_DIR / "results.json"

# ---------------- LOAD DATA ----------------
data = pd.read_csv(DATA_PATH, sep=";")
X = data.drop("quality", axis=1)
y = data["quality"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 🔴 EXPERIMENT-SPECIFIC CODE HERE
# ===============================

from sklearn.tree import DecisionTreeRegressor

EXP_ID = "EXP-04"
MODEL_NAME = "Decision Tree (depth=10 + FS)"

corr = data.corr()["quality"].abs()
selected = corr[corr > 0.15].index.drop("quality")

X_fs = data[selected]

X_train, X_test, y_train, y_test = train_test_split(
    X_fs, y, test_size=0.2, random_state=42
)

X_train_proc = X_train
X_test_proc = X_test

model = DecisionTreeRegressor(max_depth=10, random_state=42)





# ---------------- TRAIN ----------------
model.fit(X_train_proc, y_train)

# ---------------- EVAL ----------------
y_pred = model.predict(X_test_proc)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------- SAVE ----------------
joblib.dump(model, MODEL_PATH)

results = {
    "experiment": EXP_ID,
    "model": MODEL_NAME,
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("Name:", NAME)
print("Roll:", ROLL)
print("MSE:", mse)
print("R2:", r2)
