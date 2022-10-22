from keras.models import Sequential

class SequeClassifier():
    def __init__(self, units):
        self.units = units
        self.model = None
    def LSTM_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=True))
        self.model.add(LSTM(self.units))
        self.model.add(Dense(1, activation='sigmoid'))