import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Download stock data for Apple from the start of 2020 to the present
apple_data = yf.download('AAPL', start='2020-01-01')

# Print the data to ensure it's loaded
#print(apple_data.head())

apple_data.to_csv("stock_data/AAPL_stock_data.csv")

file_path = 'stock_data/AAPL_stock_data.csv'
stock_data = pd.read_csv(file_path)

print(stock_data.head())
print(stock_data.describe())
print(stock_data.info())

stock_data['Moving_Average'] = stock_data['Close'].rolling(window=20).mean()
# Add more features as needed

scaler = MinMaxScaler(feature_range=(0, 1))
stock_data['Scaled_Close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))
