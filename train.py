import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
df = pd.read_csv('heart.csv')
train = df.values
np.random.shuffle(train)
x_train = train[:, :-1]
x_train[:, 0] = x_train[:, 0]/100
x_train[:, 3] = x_train[:, 3]/100
x_train[:, 4] = x_train[:, 4]/100
x_train[:, 7] = x_train[:, 7]/100
x_train[:, 9] = x_train[:, 9]/100
y_train = train[:, -1]
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 2)
print(y_train)


model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(13,)))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=1500,
                    verbose=1)
loss = history.history['loss']
plt.plot(loss)
plt.show()
