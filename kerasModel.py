import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
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


# Use Early-Stopping
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, verbose=0, mode='auto')

model = Sequential()
#model.add(Dense(15, input_dim=30,kernel_initializer='glorot_uniform',bias_regularizer=keras.regularizers.l1_l2(1)))
model.add(Dense(400, input_dim=30, activation='linear'))
model.add(LeakyReLU(alpha=0.3))
#model.add(Dense(5,kernel_initializer='glorot_uniform'))
model.add(Dense(1,kernel_initializer='glorot_uniform', activation='linear'))

optimizer = keras.optimizers.Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)

op2 = keras.optimizers.SGD(lr=0.03, momentum=0.8, decay=0.001, nesterov=False)
op3 = keras.optimizers.RMSprop(lr=0.03, rho=0.9, epsilon=None, decay=0.0)


model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adadelta(
        lr=1.0,
        rho=0.85, 
        epsilon=None, 
        decay=0.001),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=1000, batch_size=1024,validation_data=(X_test, y_test),callbacks=[callback_early_stopping])
plt.figure(1)  
print history.history.keys()
# summarize history for accuracy  

probs = model.predict_classes(X_test)
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

# evaluate the model
scores = model.evaluate(X, Y_arr)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))