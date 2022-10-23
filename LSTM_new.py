import tensorflow as tf


def LSTM_model(input_length):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True, input_shape=(input_length, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model
