import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import sklearn
from keras import optimizers
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from scipy.io import arff
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

data = arff.loadarff('TrainingDataset.arff')
filedata = pd.DataFrame(data[0])

X = filedata.drop(['Result'],axis=1).values
Y = filedata.filter(['Result'],axis=1)


scaler = MinMaxScaler()
X_scale = scaler.fit(X)
X = scaler.transform(X)

scaler = MinMaxScaler()
Y_scale = scaler.fit(Y)
Y = scaler.transform(Y)
Y_arr = Y.ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_arr, test_size=0.33, random_state=42)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y_arr, test_size=0.33, random_state=15)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y_arr, test_size=0.33, random_state=100)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y_arr, test_size=0.33, random_state=2)


x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, Y_arr, test_size=0.3, random_state=0)

# Define model
model = keras.models.Sequential()
model.add(keras.layers.normalization.BatchNormalization(input_shape=tuple([x_train.shape[1]])))
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.core.Dense(1,   activation='sigmoid'))
model.compile(loss="mean_squared_error", optimizer="adadelta",metrics=["accuracy"])
print(model.summary())

# Use Early-Stopping
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, verbose=0, mode='auto')

# Train model
history = model.fit(x_train, y_train, batch_size=1024, epochs=1000, validation_data=(x_valid, y_valid), verbose=1, callbacks=[callback_early_stopping])


# evaluate the model
scores = model.evaluate(x_valid, y_valid)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



plt.figure(1)  
probs = model.predict_classes(x_valid)
preds = probs.ravel()

plt.subplot(121)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

# summarize history for loss  

plt.subplot(122)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  