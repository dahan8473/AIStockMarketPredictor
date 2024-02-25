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

#importing data file, must be in csv format
dataset = pd.read_csv("""insert file name as string""")
"""For csv dataset, the dependable variable is price and there should be more than 1 parameters for independent variable"""
x = dataset.drop(columns=["""insert column name containing result (i.e. the dependent variable results) here"""])#these columns include data for independent variables hiding the result, to train the AI
y = dataset["""insert column name containing result (i.e. the dependent variable results) here"""]#this is the column containing the results


from sklearn.model_selection import train_test_split #this module separates the data set into training part and actual data part, useful for training and validating data set
x_train, y_train, x_test, y_test = train_test_split(x,y, test_size= 0.2)

model = tf.keras.model.Sequential()#this is the model provided, most likely this model isn't perfect for stock but i'll replace it later
model.add(tf.keras.layers.Dense(512, input_shape = x_train.shape[1:], activation = 'sigmoid')) #input layer of neural network
model.add(tf.keras.layers.Dense(512, activation = 'sigmoid'))#processing layer
model.add(tf.keras.layers.Dense(3, activation = 'sigmoid'))#output layer, for now will be either increase, decrease, or stay same

model.compile(optimzer="adam", loss="mean_squared_error", metrics = ["accuracy"])#according to my research these optimizer and loss algorithms are best for stocks

model.fit(x_train, y_train, epochs = 5000)#training

model.evaluate(x_test,y_test)#actual data evaluation


