import yfinance as yf

# Download stock data for Apple from the start of 2020 to the present
apple_data = yf.download('AAPL', start='2020-01-01')

# Print the data to ensure it's loaded
print(apple_data.head())
