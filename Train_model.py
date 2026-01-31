import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ================= LOAD DATA =================
df = pd.read_csv(
    r"E:/Electric Vehicle Popolation Data/Data/Electric_Vehicle_Population_Data.csv"
)
df.columns = df.columns.str.strip()

df = df[
    [
        "Model Year",
        "Electric Vehicle Type",
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
        "Electric Range"
    ]
].dropna()

# ================= TARGET =================
df["Long_Range_EV"] = (df["Electric Range"] >= 250).astype(int)

y = df["Long_Range_EV"]
X = df.drop(["Long_Range_EV", "Electric Range"], axis=1)

# ================= ENCODE & SCALE =================
X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ================= TRAIN =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# ================= SAVE FILES =================
os.makedirs("model", exist_ok=True)

with open("model/ev_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/ev_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/ev_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, scaler and features saved in model/ folder")
