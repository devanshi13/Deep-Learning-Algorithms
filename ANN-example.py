# Part 1 - Data Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:,3:13]
Y = df.iloc[:,13]

#create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#concatenate datadrames
X = pd.concat([X,geography,gender], axis=1)

#drop unnecessary variables
X = X.drop(['Geography','Gender'], axis=1)

#split the dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=0)

#feature scaling
#feature scaling improves the training of model as all the inputs are scaled to a range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Create ANN Model

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

#initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

# Add the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compile the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, Y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)

# list all data in history

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

classifier.summary()
#total params are total number of weights and biases 

# Part 3 - Make the predictions and evaluate the model

# Predict the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,Y_test)

#Conclusions
#1.For hidden layers we have used relu actiavtion function as it works better 
#2.For initialization he_uniform works well with relu while for sigmoid we use glorot_uniform
#3.In output we have only two categories so loss fn. is binary_crossentropy
#4.If we increase number of hidden units in every layer then accuracy was almost same
#5.If we increase number of hidden layers, accuracy is almost same
#6.If we apply dropout then accuracy gets decreased, which means we shout dropout only in case of very deep neural network
#7. classifier.add(Dropout(p)) p=dropout ratio



