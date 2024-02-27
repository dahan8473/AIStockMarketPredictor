"""
========================================================================================================
Name: Stock predictor AI
Date last modified: 2024/02/26
Authors: David Liu, Justin Oh
Description:
Required dependencies: 
========================================================================================================
"""

#importing necessary libraries
import yfinance as yf
import pandas as pd
import ai_training_module as ai
from sklearn.preprocessing import MinMaxScaler
import eda



# Download stock data for Apple from the start of 2020 to the present
company = input("Enter company code: ")
company_data = yf.download(company, start='2020-01-01')

# Print the data to ensure it's loaded
#print(apple_data.head())

company_data.to_csv("stock_data/stock_data_initial.csv")

file_path = 'stock_data/stock_data_initial.csv'
stock_data = pd.read_csv(file_path)

print(stock_data.head())
print(stock_data.describe())
print(stock_data.info())

stock_data['Moving_Average'] = stock_data['Close'].rolling(window=20).mean()
# Add more features as needed

scaler = MinMaxScaler(feature_range=(0, 1))
stock_data['Scaled_Close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))


eda.convertDate(stock_data)
eda.plotThis(stock_data, "Close", company)
predictions = ai.getPrediction()



print("The predictions are: %s for the next %d days" % (predictions, len(ai.LOOKUP)))



