from keras.models import Sequential
from tensorflow.python.keras.layers import Dropout

def LSTM_model(input_length):

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(input_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model
