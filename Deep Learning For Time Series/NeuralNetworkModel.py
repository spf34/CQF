import numpy as np
import pandas as pd

from keras import models
from keras.layers import Dense, Dropout, LSTM
from keras import metrics

'''
To note - the classes in this file are closely based on:
https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/core/

The main differences are: 
  - use of dictionary for model inputs in NeuralNetworkModel instead of json
  - update of DataLoader for classification task
  - extra methods added
'''


class NeuralNetworkModel(object):
    """Create Neural Networks using Keras"""

    def __init__(self, model_specification, nn_type, name):
        self.model = models.Sequential()
        self.model_specs = model_specification
        self.nn_type = nn_type
        self.name = name

    def build_model(self):
        loss = self.model_specs['loss']
        optimiser = self.model_specs['optimiser']

        i = 1
        while i in self.model_specs:
            layer = self.model_specs[i]
            neurons = layer['neurons'] if 'neurons' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'].lower() == 'dense':
                if i == 1:
                    self.model.add(Dense(neurons, activation=activation, input_shape=(input_dim,)))
                else:
                    self.model.add(Dense(neurons, activation=activation))
            if layer['type'].lower() == 'lstm':
                self.model.add(
                    LSTM(units=neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'].lower() == 'dropout':
                self.model.add(Dropout(dropout_rate))

            i += 1

        self.model.compile(loss=loss, optimizer=optimiser, metrics=[metrics.binary_accuracy])

    def get_basic_trained_model(self, x_train, y_train, epochs=5, batch_size=512, x_val=None, y_val=None):
        self.build_model()

        if (x_val is None or y_val is None):
            history = self.model.fit(x_train, y_train,
                                     epochs=epochs,
                                     batch_size=batch_size)

        else:
            history = self.model.fit(x_train, y_train,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_data=(x_val, y_val))
        return history

    def get_model_scores(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(f'Average Prediction: {predictions.mean()}')
        pd.DataFrame(predictions, columns=['sigmoid']).to_csv(
            f'results/{self.nn_type}/predictions/{self.name}.csv')
        predictions[predictions > 0.5] = 1
        predictions[predictions < 0.5] = 0
        return (2 * predictions - 1 == y_test).mean(), predictions.mean()

    def save_summary(self, nn_type, accuracy, avg_pred):
        f = open(f'results/{nn_type}/model_summary/{self.name}_summary.txt', 'w')
        self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'accuracy: {accuracy}\n')
        f.write(f'average prediction: {avg_pred}')
        f.close()


class DataLoader(object):
    """Format data to be fed directly into model"""

    def __init__(self, feature_data, test_train_split):
        """

        Args:
            feature_data (pd.DataFrame):
                IMPORTANT - the final column must contain the quantity to forecast and no other!

            test_train_split (float):
                proportion of data to be reserved for training
        """
        split_idx = int(len(feature_data) * test_train_split)

        cols = feature_data.columns.to_list()

        self.training_data = feature_data.get(cols).values[:split_idx]
        self.test_data = feature_data.get(cols).values[split_idx:]

        self.len_train = len(self.training_data)
        self.len_test = len(self.test_data)
        self.len_train_windows = None

    def get_shaped_data(self):
        pass

    def get_lstm_train_data(self, seq_len):
        """
        Create input feature, output training data windows
        """
        input = []
        output = []
        for i in range(self.len_train - seq_len):
            x, y = self.get_next_window(i, seq_len)
            input.append(x)
            output.append(y)
        return np.array(input), np.array(output)

    def get_train_data(self):
        return self.training_data[:, :-1], self.training_data[:, -1]

    def get_test_data(self):
        return self.test_data[:, :-1], self.test_data[:, -1]

    def get_lstm_test_data(self, seq_len):
        """
        Create input feature, output test data windows
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.test_data[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)

        x = data_windows[:, :, :-1]
        y = data_windows[:, -1, [-1]]
        return x, y

    def get_next_window(self, i, seq_len):
        """
        Generates the next data window from the given index location i
        """
        window = self.training_data[i:i + seq_len]
        x = window[:, :-1]
        y = window[-1, [-1]]

        return x, y

    def get_validation_data(self, X_train, y_train, validation_size):
        """
        Split training data into train and validation set
        """
        idx = int(len(X_train) * (1 - validation_size))
        X_val, y_val = X_train[idx:], y_train[idx:]
        X_train, y_train = X_train[:idx], y_train[:idx]
        return X_train, X_val, y_train, y_val
