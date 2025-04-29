
import yfinance as yf
import pandas as pd
data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

import ta

from ta.momentum import RSIIndicator #
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = yf.download("AAPL", start="2023-1-1",end="2025-4-25")
data.dropna(inplace=True)
# Check the original data
print(data.head())
print(data.tail())
print(data.describe())

# Rename columns to remove 'AAL' for simple'
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Check the data with renamed columns
print(data.head())
print(data['Close'])

# Feature Engineering
# For technical  analysis there are  4 most important indicator: RSI, SMA, MACD, Bollinger band

# Adding RSI
# RSI = 100 - [100 / (1 + RS)]
# RS (Relative Strength) = Average of 'n' days' up closes / Average of 'n' days' down closes.
# n = 14 (default period).

rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi.rsi()

# if RSI>70, the stock is overbought, signal to sell in short term.
# if RSI<30, oversold, buy signal for short term.

# Adding 20 days and 50 days Moving Averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average

# 1, If Close price below SMA-20, short term buy signal;
# If price above SMA-20, short term sell signal, the higher the price, sooner we should sell.
# 2, when SMA20 Cross above SMA50, golden cross, buy signal.
# when SMA20 cross below SMA50, death cross, sell signal.

# Show the data with the added features
print(data[['Close', 'SMA_20', 'SMA_50', 'RSI']].tail(10))

# Define buy and sell signals based on RSI
data['Buy_Signal'] = (data['RSI'] < 30)
data['Sell_Signal'] = (data['RSI'] > 70)

# SMA crossover buy/sell signals
data['SMA_Golden_Cross'] = (data['SMA_20'] > data['SMA_50']) & (data['SMA_20'].shift(1) <= data['SMA_50'].shift(1))
data['SMA_Death_Cross'] = (data['SMA_20'] < data['SMA_50']) & (data['SMA_20'].shift(1) >= data['SMA_50'].shift(1))

data.dropna(inplace=True)
#show the plot  of RSI and SMA by seaborn
sns.set(style="darkgrid", context="talk")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Price chart SMA-20,SMA-50,
ax1.plot(data.index, data['Close'], label='Close', color='blue',linewidth=1)
ax1.plot(data.index, data['SMA_20'], label='SMA 20', color='red', linestyle='--',linewidth=1)
ax1.plot(data.index, data['SMA_50'], label='SMA 50', color='green', linestyle='-.',linewidth=1)

ax1.plot(data[data['SMA_Golden_Cross']].index, data[data['SMA_Golden_Cross']]['Close'],
         'o', color='gold', markersize=6, label='Golden Cross')
ax1.plot(data[data['SMA_Death_Cross']].index, data[data['SMA_Death_Cross']]['Close'],
         'x', color='red', markersize=6, label='Death Cross')

ax1.set_title('Figure.1 AAPL Price with SMA20 & SMA50')
ax1.set_ylabel('Price ($)')
ax1.legend()

# RSI chart
ax2.plot(data.index, data['RSI'], label='RSI', color='navy',linewidth=1)
ax2.axhline(70, color='orange', linestyle='--', alpha=0.8, linewidth=2)
ax2.axhline(30, color='orange', linestyle='--', alpha=0.8, linewidth=2)

ax2.plot(data[data['Buy_Signal']].index, data[data['Buy_Signal']]['Close'],
         '^', markersize=8, color='green', label='Buy Signal (RSI < 30)')
ax2.plot(data[data['Sell_Signal']].index, data[data['Sell_Signal']]['Close'],
         'v', markersize=8, color='red', label='Sell Signal (RSI > 70)')

ax2.set_title('Figure.2 RSI with Sell and Buy signal')
ax2.set_ylabel('RSI Value')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.show()

# Add MACD indicator: Calculate MACD

ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Hist'] = data['MACD'] - data['Signal_Line']

data.dropna(inplace=True)
# Plot MACD
# Identify buy and sell signals (MACD crossover)
data['Buy_Signal'] = (data['MACD'] > data['Signal_Line']) & (data['MACD'].shift(1) <= data['Signal_Line'].shift(1))
data['Sell_Signal'] = (data['MACD'] < data['Signal_Line']) & (data['MACD'].shift(1) >= data['Signal_Line'].shift(1))

ticker = "AAPL"

plt.figure(figsize=(14, 6))
plt.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=1.5)
plt.plot(data.index, data['Signal_Line'], label='Signal Line', color='red', linewidth=1.5)
plt.bar(data.index, data['MACD_Hist'], label='Histogram', color='green', alpha=0.5)

# Buy/Sell markers
plt.plot(data[data['Buy_Signal']].index, data['MACD'][data['Buy_Signal']],
         '^', color='green', markersize=10, label='Buy Signal')
plt.plot(data[data['Sell_Signal']].index, data['MACD'][data['Sell_Signal']],
         'v', color='red', markersize=10, label='Sell Signal')

plt.title(f"{ticker} MACD Indicator with Buy/Sell Signals (Seaborn Style)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("MACD Value")
plt.legend()
plt.tight_layout()
plt.show()

#Bollinger Bands added, predict price trend and visualize volility.

# Calculate Bollinger Bands
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['STD_20'] = data['Close'].rolling(window=20).std()
data['Upper_Band'] = data['SMA_20'] + (2 * data['STD_20'])
data['Lower_Band'] = data['SMA_20'] - (2 * data['STD_20'])

data.dropna(inplace=True)
# Plot the closing price along with Bollinger Bands
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.6)
plt.plot(data['SMA_20'], label='SMA 20', color='red', alpha=0.8)
plt.plot(data['Upper_Band'], label='Upper Band', color='green', linestyle='--')
plt.plot(data['Lower_Band'], label='Lower Band', color='green', linestyle='--')

# Adding titles and labels
plt.title('Figure.3 Bollinger Bands')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# ML Model
# Feature Engineering - Calculate Indicators

# Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# MACD
def compute_macd(data):
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    return data

# Bollinger Band
def compute_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    data['BB_upper'] = sma + (2 * std)
    data['BB_lower'] = sma - (2 * std)
    return data

# Apply indicators
data = compute_rsi(data)
data = compute_macd(data)
data = compute_bollinger_bands(data)

# Drop rows with NaNs from indicators
data.dropna(inplace=True)

# Feature columns (add more as needed)
features = [ 'Open','High', 'Low', 'Close', 'Volume',
             'RSI', 'MACD', 'Signal_Line', 'MACD_Hist',
             'BB_upper', 'BB_lower']

X = data[features]
y = data['Close']# Predict next day's close

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)




for n in [10, 50, 100, 200, 500]:
    model = RandomForestRegressor(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"n_estimators={n}, RMSE={rmse:.2f}")

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)

# Plot Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--')
plt.legend()
plt.title('AAPL Stock Price Prediction (2 Months Ahead)')
plt.show()

# Assuming model has already been trained and y_test and y_pred are available
# y_test: Actual values (ground truth)
# y_pred: Predicted values by the model

# Calculate the evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Display the results
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100}%")


# Predict for the next day price (using the most recent data point)

latest_data = data.iloc[-1][features].values.reshape(1, -1)
latest_data_df = pd.DataFrame(latest_data, columns=features)

predicted_price = model.predict(latest_data_df)
print(f" RandomForest Predicted AAPL Price Next Day: {predicted_price[0]}")

# Plot feature importance and model simplification

# Get feature importances from the trained model
importances = model.feature_importances_
feature_names = X.columns  # Ensure X is a DataFrame with column names

# Create a DataFrame for easier plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


# Print the scores
print("Feature Importance Scores:\n")
print(importance_df.to_string(index=False))

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df,hue="Feature", palette='viridis', legend=False)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Create and train the SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svr_model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Plot Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--')
plt.title('AAPL Stock Price Prediction using Support Vector Machine')
plt.legend()
plt.show()

# Predict for the next day (use the latest data point)
latest_data = data.iloc[-1][features].values.reshape(1, -1)
predicted_price_next_day = svr_model.predict(latest_data)
print(f"SVM model Predicted AAPL price for next day: {predicted_price_next_day[0]}")

# Assuming the SVR model is already trained
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)

# Coefficients (feature importance)
feature_importance = np.abs(svr_model.coef_)
for feature, importance in zip(X.columns, feature_importance[0]):
    print(f" SVR Feature: {feature}, Importance: {importance}")

# Try XGboost model

# Assuming `data` contains the necessary features (like Close, High, Low, etc.)

# Features and target
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist', 'BB_upper', 'BB_lower']
target = 'Close'  # Predicting the Close price

# Prepare data
X = data[features]
y = data[target]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Plot Actual vs Predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red', linestyle='--')
plt.legend()
plt.title('AAPL Stock Price Prediction with XGBoost')
plt.show()

# Feature Importance Plot
xgb.plot_importance(model, importance_type='weight', max_num_features=10, height=0.8)
plt.title('Feature Importance (Top 10 features)')
plt.show()

# Get feature importance from the XGBoost model
importance_scores = model.feature_importances_

# Create a dictionary of feature names and their importance
importance_dict = dict(zip(features, importance_scores))

# Sort features by importance score (highest first)
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print sorted feature importance
print("\nFeature Importance (sorted):")
for feature, score in sorted_importance:
    print(f"{feature}: {score:.4f}")

# Predict for the next day (use the latest data point)
latest_data = data.iloc[-1][features].values.reshape(1, -1)
predicted_price_next_day = model.predict(latest_data)
print(f"XGboost model Predicted AAPL price for next day: {predicted_price_next_day[0]}")

# if predicting the price , the original feature is significant important than indicator
#so indicate works as trend direction.


