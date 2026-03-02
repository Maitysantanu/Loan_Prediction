from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load trained full pipeline model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("frontend.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert numeric fields
        numeric_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]

        for col in numeric_columns:
            data[col] = float(data[col])

        df = pd.DataFrame([data])

        proba = model.predict_proba(df)[0][1]

        prediction = "Loan Approved" if proba > 0.6 else "Loan Not Approved"

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(float(proba), 3),
            previous_values=data
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)