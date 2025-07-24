from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# Load model
model = joblib.load("xgboost_fraud_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Extract raw inputs
        filing_year = float(data['Filing_Year'])
        assets = float(data['Assets'])
        revenues = float(data['Revenues'])
        net_income = float(data['NetIncomeLoss'])
        op_expenses = float(data['OperatingExpenses'])
        op_income = float(data['OperatingIncomeLoss'])
        debt = float(data['Debt'])
        liabilities = float(data['Liabilities'])
        equity = float(data['StockholdersEquity'])
        cash = float(data['CashAndCashEquivalentsAtCarryingValue'])
        gross_profit = float(data['GrossProfit'])
        curr_assets = float(data['AssetsCurrent'])
        noncurr_assets = float(data['AssetsNoncurrent'])

        current_ratio = float(data['Current_Ratio'])
        debt_ratio = float(data['Debt_Ratio'])
        profit_margin = float(data['Profit_Margin'])
        return_on_assets = float(data['Return_on_Assets'])
        operating_margin = float(data['Operating_Margin'])
        equity_ratio = float(data['Equity_Ratio'])
        cash_ratio = float(data['Cash_Ratio'])

        # Safe transforms
        def safe_log(x):
            return np.log1p(x) if x > 0 else 0

        clipped_current_ratio = np.clip(current_ratio, 0, 10)
        log_debt_ratio = safe_log(debt_ratio)
        log_profit_margin = safe_log(profit_margin)
        log_return_on_assets = safe_log(return_on_assets)
        log_operating_margin = safe_log(operating_margin)
        log_equity_ratio = safe_log(equity_ratio)
        log_cash_ratio = safe_log(cash_ratio)

        # Final feature array
        features = np.array([[
            filing_year, assets, revenues, net_income, op_expenses,
            op_income, debt, liabilities, equity, cash, gross_profit,
            curr_assets, noncurr_assets, clipped_current_ratio,
            log_debt_ratio, log_profit_margin, log_return_on_assets,
            log_operating_margin, log_equity_ratio, log_cash_ratio
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Fraudulent" if prediction == 1 else "Not Fraudulent"

        # Generate synthetic training data (for LIME to work properly)
        np.random.seed(42)
        synthetic_data = features + np.random.normal(0, 0.1, size=(100, features.shape[1]))

        explainer = LimeTabularExplainer(
            training_data=synthetic_data,
            feature_names=[
                "Filing Year", "Assets", "Revenues", "Net Income", "Operating Expenses",
                "Operating Income", "Debt", "Liabilities", "Equity", "Cash", "Gross Profit",
                "Current Assets", "Non-Current Assets", "Current Ratio",
                "Debt Ratio", "Profit Margin", "Return on Assets", "Operating Margin",
                "Equity Ratio", "Cash Ratio"
            ],
            class_names=["Not Fraudulent", "Fraudulent"],
            mode='classification'
        )

        exp = explainer.explain_instance(features[0], model.predict_proba, num_features=3)
        lime_explanation = [f"{feat} {'↑' if weight > 0 else '↓'} (impact: {abs(weight):.4f})"
                            for feat, weight in exp.as_list()]

        explanation_text = "Top factors influencing this prediction: " + ", ".join(lime_explanation)

        return render_template("index.html", prediction_text=f"Prediction: {result}", 
                               explanation=explanation_text, 
                               values=data)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
