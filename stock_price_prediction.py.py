import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Download stock data
df = yf.download("AAPL", start="2018-01-01", end="2023-12-31", auto_adjust=False)

# Step 2: Feature Engineering
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Close'].rolling(window=10).std()
df.dropna(inplace=True)

# Step 3: Features & Target
X = df[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Volatility']]
y = df[['Close']]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)

# Step 5: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, np.ravel(y_train))

# Step 6: Predictions
predictions = model.predict(X_test)

# Step 7: Evaluation
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("📊 RMSE:", round(rmse, 2))
print("📈 R² Score:", round(r2, 2))

# Step 8: Save predictions to CSV
results_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual Price': y_test.values.flatten(),
    'Predicted Price': predictions
})
results_df.to_csv("predictions_output.csv", index=False)
print("✅ Results saved to predictions_output.csv")

# Step 9: Bar Graph with smart Indian formatting
N = 30
indices = np.arange(N)
bar_width = 0.4

dates = results_df['Date'].dt.strftime('%d-%b-%y').values[:N]
actual = results_df['Actual Price'][:N]
predicted = results_df['Predicted Price'][:N]

plt.figure(figsize=(15, 7))
plt.bar(indices, actual, width=bar_width, color='blue', label='Actual Price')
plt.bar(indices + bar_width, predicted, width=bar_width, color='orange', label='Predicted Price')

plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (in ₹)", fontsize=12)
plt.title("AAPL Stock Price Prediction - Intelligent ₹ Formatting", fontsize=14)
plt.xticks(indices + bar_width / 2, labels=dates, rotation=45)
plt.legend()
plt.grid(True, axis='y')

# ✅ Smart Indian currency formatter
def smart_inr_format(value, _):
    if value >= 1e7:
        return f'{value/1e7:.2f} Cr'
    elif value >= 1e5:
        return f'{value/1e5:.2f} L'
    elif value >= 1e3:
        return f'{value/1e3:.0f}K'
    else:
        return f'{value:.0f}'

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(smart_inr_format))
plt.tight_layout()
plt.show()
