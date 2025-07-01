# Stock-Price-Prediction
Predict stock prices using real-time data, ML models, and beautiful Indian-style bar graphs — built in Python with zero deep learning dependencies!
# 📈 Stock Price Prediction using Machine Learning (Random Forest)

This project predicts stock prices using real-time historical data and the **Random Forest Regressor** model — with clean visualizations, Indian currency formatting (Lakhs/Crores), and a complete CSV export of results.

---

## 🔍 Problem Statement

Stock market prices are volatile and difficult to predict using traditional methods. Investors struggle to make informed decisions due to lack of accurate, automated forecasting tools.

---

## ✅ Solution Overview

We used **machine learning algorithms** to predict stock prices using historical features like:
- Opening, High, Low, Volume
- Moving Averages (10-day, 50-day)
- Price Volatility

This project uses **real-time data from Yahoo Finance**, trains a model, and visualizes actual vs predicted prices using bar graphs (in ₹ Lakhs/Crores).

---

## 🛠 Tech Stack

| Component     | Tool / Library            |
|---------------|---------------------------|
| Language      | Python 3.13.5             |
| Data Source   | Yahoo Finance via `yfinance` |
| Model         | Random Forest Regressor (`sklearn`) |
| Visualization | `matplotlib`              |
| Other Tools   | `pandas`, `numpy`, `statsmodels` |

---

## 📊 Output Example

- Blue bars = Actual stock price  
- Orange bars = Predicted price  
- Prices shown in **₹ Lakhs/Crores**  
- Graph x-axis shows **real historical dates**

> 📁 You can find the predictions inside: `predictions_output.csv`

![Bar Graph Output Screenshot](sample_graph.png)

---

## 📂 Project Files

| File                    | Description                                   |
|-------------------------|-----------------------------------------------|
| `stock_price_prediction.py` | Main Python script                        |
| `predictions_output.csv`    | Actual vs Predicted price output (CSV)    |
| `sample_graph.png`          | Screenshot of graph (optional)            |
| `README.md`                | Project documentation                      |

---

## ⚙️ How to Run This Project

1. Clone the repository  
2. Make sure Python 3.13+ is installed  
3. Install the required libraries:
   ```bash
   pip install yfinance pandas numpy matplotlib scikit-learn statsmodels
