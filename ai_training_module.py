"""
Preliminary AI training module
Currently runs on linear regression and neural network algorithm, but for stock application we must use Long Short Term Memory algorithm,
which i will figure out later.
This is because stock depends on so many factors and result for the prediction is not just simply binary.
Use pip install tensorflow and pip install sklearn to install required dependencies
"""

#modules required
import sklearn.model_selection 
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler#for scaling purpose
from collections import deque

#importing data file, must be in csv format
dataset = pd.read_csv("stock_data/stock_data_initial.csv")
#scale data
scalar = MinMaxScaler()
dataset['Close'] = scalar.fit_transform(np.expand_dims(dataset['Close'].values, axis=1))


#settings
SEQ_LENGTH = 7 #how many days of data do we want to analyze?
LOOKUP = [1,2,3]#1 = tomorrow, 2 = after tomorrow, 3 = 3 days from now


dataset = dataset.drop(columns=["High", "Low", "Adj Close", "Volume", "Open" ])
"""For csv dataset, the dependable variable is price and there should be more than 1 parameters for independent variable"""
x = dataset.drop(columns=["Close"])#these columns include data for independent variables hiding the result, to train the AI
y = dataset["Close"]#this is the column containing the results



#preprocessing, we will "shift" the data here
def preprocessData(days):#days is a parameter to see how many days you want the prediction for
    data = dataset.copy()
    data["Future"] = dataset["Close"].shift(-days)
    last_sequence = np.array(data[["Close"]].tail(days))
    data.dropna(inplace=True)
    sequence_list = []
    sequence = deque(maxlen=SEQ_LENGTH)

    for start, target in  zip(data[['Close'] + ["Date"]].values, data['Future'].values):
        sequence.append(start)
        if len(sequence) == SEQ_LENGTH:
            sequence_list.append([np.array(sequence),target])

    last_sequence = list([s[:len(['close'])] for s in sequence]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)#convert to compatible np array bc tensorflow is such BS and won't accept regular array
    
    X,Y = [], []
    for seq, target in sequence_list:
        X.append(seq)
        Y.append(target)

    X = np.array(X)
    Y = np.array(Y)

    return data, last_sequence, X, Y





#from sklearn.model_selection import train_test_split #this module separates the data set into training part and actual data part, useful for training and validating data set
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)



def getModel(x_train, y_train):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, return_sequences = True, input_shape = (SEQ_LENGTH, len(["Close"]))))#LSTM layer 1
    model.add(tf.keras.layers.Dropout(0.3))#hidden layer

    model.add(tf.keras.layers.LSTM(256, return_sequences = False, input_shape = (SEQ_LENGTH, len(["Close"]))))#LSTM Layer 2
    model.add(tf.keras.layers.Dropout(0.3))#hidden layer

    model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape[1:], activation = 'sigmoid')) #output Dense layer

    model.add(tf.keras.layers.Dense(1))#final result layer


    model.compile(optimizer ='adam', loss="mean_squared_error", metrics = ["accuracy"])#according to my research these optimizer and loss algorithms are best for stocks

    model.fit(x_train, y_train, epochs = 80)#training
    model.summary()
    return model
#model.evaluate(x_test,y_test)#actual data evaluation

def getPrediction():
    predictions = []

    for days in LOOKUP:
        data, last_sequence, x_train, y_train = preprocessData(days)
        x_train = x_train[:,:,:len(['Close'])].astype(np.float32)

        model = getModel(x_train,y_train)

        last_sequence = last_sequence[-SEQ_LENGTH:]
        last_sequence = np.expand_dims(last_sequence, axis = 0)
        prediction = model.predict(last_sequence)
        predicted_price = scalar.inverse_transform(prediction)[0][0]

        predictions.append(round(float(predicted_price),2))

    return predictions

