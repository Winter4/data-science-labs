import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

dataframe = pd.read_csv("car.data")

X = dataframe.values[:, :-1]
Y = dataframe.values[:, -1]

print(dataframe)

X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
X[:, 1] = LabelEncoder().fit_transform(X[:, 1])
X[:, 2] = LabelEncoder().fit_transform(X[:, 2])
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X[:, 4] = LabelEncoder().fit_transform(X[:, 4])
X[:, 5] = LabelEncoder().fit_transform(X[:, 5])
Y = LabelEncoder().fit_transform(Y)

X = X.astype('float32')

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

Y_train = tf.keras.utils.to_categorical(Y_train)
Y_test = tf.keras.utils.to_categorical(Y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
history = model.fit(X_train, Y_train, epochs = epochs, batch_size = 5)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Точность на тестовой выборке: {:.2f}%".format(accuracy * 100))

plt.plot(
    np.arange(1, epochs + 1), 
    history.history['accuracy'], label='Точность'
)
plt.plot(
    np.arange(1, epochs + 1), 
    history.history['loss'], label='Функция потерь'
)
plt.xlabel('Эпохи', size=14)
plt.legend()
plt.show()