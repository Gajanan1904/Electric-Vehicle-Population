from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")

# ================= LOAD MODEL FILES =================

model = pickle.load(open("model/ev_model.pkl", "rb"))
scaler = pickle.load(open("model/ev_scaler.pkl", "rb"))
features = pickle.load(open("model/ev_features.pkl", "rb"))

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    user_input = dict.fromkeys(features, 0)

    user_input["Model Year"] = float(request.form["model_year"])

    if request.form["ev_type"] == "PHEV":
        col = "Electric Vehicle Type_Plug-in Hybrid Electric Vehicle (PHEV)"
        if col in user_input:
            user_input[col] = 1

    if request.form["cafv"] == "Eligible":
        col = "Clean Alternative Fuel Vehicle (CAFV) Eligibility_Eligible"
        if col in user_input:
            user_input[col] = 1

    input_array = np.array([user_input[col] for col in features]).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    prob = model.predict_proba(scaled_input)[0][1] * 100
    label = "🚗⚡ Long-Range EV" if prob >= 50 else "🔋 Short-Range EV"

    return render_template(
        "index.html",
        prediction_text=f"{label} ({prob:.2f}%)",
        confidence=round(prob, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
