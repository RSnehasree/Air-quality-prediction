# import os
# import joblib
# import pandas as pd
# from flask import Flask, render_template, request

# # Initialize Flask app
# app = Flask(__name__)

# # Load trained model
# MODEL_PATH = "stacking_model.pkl"
# model = joblib.load(MODEL_PATH)

# # Feature columns (must match training pipeline order)
# FEATURE_COLUMNS = [
#     "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
#     "Temperature", "Humidity", "WindSpeed",
#     "RespiratoryCases", "CardiovascularCases",
#     "HospitalAdmissions", "HealthImpactScore"
# ]

# # Updated health class mapping (Option B style + Color Code Support)
# HEALTH_IMPACT_LABELS = {
#     0: ("Class 0 – Safe", "No health risk.", "green"),
#     1: ("Class 1 – Mild Impact", "Minor irritation for sensitive groups.", "yellow"),
#     2: ("Class 2 – Moderate Impact", "Respiratory symptoms likely for public.", "orange"),
#     3: ("Class 3 – High Risk", "Breathing discomfort. Limit outdoor activity.", "red"),
#     4: ("Class 4 – Severe Risk", "Serious risk! Avoid outdoor exposure!", "purple")
# }

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     color_class = ""

#     if request.method == "POST":
#         try:
#             # Get input values from form and convert to float
#             input_data = {col: float(request.form[col]) for col in FEATURE_COLUMNS}
            
#             # Convert to DataFrame matching model input format
#             df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

#             print("\n📌 Prediction Input Data:")
#             print(df.to_string(index=False))  # Debugging

#             # Predict class
#             prediction = int(model.predict(df)[0])

#             # Get display values (label, message, color class)
#             label, message, color_code = HEALTH_IMPACT_LABELS.get(prediction, ("Unknown", "", ""))

#             # Wrap result for UI
#             result = f"{label} – {message}"
#             color_class = f"class_{prediction}"

#         except Exception as e:
#             print("Error:", e)
#             return render_template("index.html", error="Invalid input! Please check values and try again.")

#     return render_template(
#     "index.html",
#     result=result if request.method == "POST" else None,
#     color_class=color_class,
#     feature_columns=FEATURE_COLUMNS
# )
# if __name__ == "__main__":
#     app.run(debug=True)





import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# File Paths
MODEL_PATH = "stacking_model.pkl"
SCALER_PATH = "scaler.pkl"

# Check required files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Required model or scaler file missing from workspace!")

# Load model & scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature columns (must match training order)
FEATURE_COLUMNS = [
    "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed",
    "RespiratoryCases", "CardiovascularCases",
    "HospitalAdmissions", "HealthImpactScore"
]

# Health class metadata
HEALTH_IMPACT_LABELS = {
    0: ("Class 0 – Safe", "No health risk.", "green"),
    1: ("Class 1 – Mild Impact", "Minor irritation for sensitive groups.", "yellow"),
    2: ("Class 2 – Moderate Impact", "Respiratory symptoms likely for public.", "orange"),
    3: ("Class 3 – High Risk", "Breathing discomfort. Limit outdoor activity.", "red"),
    4: ("Class 4 – Severe Risk", "Serious risk! Avoid outdoor exposure!", "purple")
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    color_class = ""

    if request.method == "POST":
        try:
            # 1. Parse inputs
            input_data = {col: float(request.form[col]) for col in FEATURE_COLUMNS}
            df_raw = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

            # 2. Scale features using stored StandardScaler (CRITICAL FIX)
            df_scaled = scaler.transform(df_raw)

            # 3. Predict class
            prediction = int(model.predict(df_scaled)[0])

            # 4. Fetch label and description
            label, message, _ = HEALTH_IMPACT_LABELS.get(prediction, ("Unknown", "", ""))
            result = f"{label} – {message}"
            color_class = f"class_{prediction}"

        except Exception as e:
            print("Prediction Error:", str(e))
            return render_template("index.html", error=f"Invalid input or prediction error: {str(e)}", feature_columns=FEATURE_COLUMNS)

    return render_template(
        "index.html",
        result=result,
        color_class=color_class,
        feature_columns=FEATURE_COLUMNS
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)