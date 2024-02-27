# eda.py or cells in eda.ipynb

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def setData(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Convert 'Date' column to datetime and set as index
def convertDate(dataset):
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset.set_index('Date', inplace=True)

# Plot closing price over time
def plotThis(dataset,label,title):
    plt.figure(figsize=(10, 5))
    plt.plot(dataset["Close"], label=label)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Display descriptive statistics
    print(dataset.describe())

    # Correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# More EDA...
