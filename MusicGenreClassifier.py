import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix


import tensorflow.keras as keras
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from keras import models
from keras import layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
# Keras
from keras import models, Sequential
from keras import layers
import numpy as np

data = pd.read_csv('data.csv')
print(data.shape)
# dropping useless column
data.drop(['filename'], axis=1, inplace=True)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
# normalizing
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape[1])
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))
model.summary()

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history= model.fit(X_train,y_train, epochs=50,batch_size=256,validation_data=(X_test,y_test))
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
results = model.evaluate(X_test, y_test)

print('Test loss:', results[0])
print('Test accuracy:', results[1])
predictions = model.predict(X_test)
print(predictions[0].shape)

cf_matrix = confusion_matrix(y_test, np.argmax(predictions,axis=1))
print(cf_matrix)
from sklearn.metrics import classification_report, confusion_matrix

print('Classification_report')
print(classification_report(y_test , np.argmax(predictions,axis=1)))
sns.heatmap(cf_matrix, annot=True)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.show()