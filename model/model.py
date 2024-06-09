import tensorflow as tf
from tensorflow.keras import Sequential, layers

def create_model():
    model=Sequential()
    model.add(layers.Flatten(input_shape=(30,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model