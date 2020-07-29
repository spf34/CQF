import os
import random
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from DataWrangler import DataWrangler
from DataExploration import DataExploration, DecisionTreeHelpers
from helpers import normalise_feature_data, create_directories, plot_network_history
from NeuralNetworkModel import NeuralNetworkModel, DataLoader
import keras

plt.style.use('ggplot')

create_directories()

tickers = ['san', 'bbva']

s_days = [1, 2, 3, 4, 5, 10, 21, 50]
l_days = [20, 40, 60, 80, 100, 200]
s5_days = [5 * x for x in s_days]

day_ranges = {'returns': s_days, 'sgn_returns': s_days, 'cds_returns': s_days, 'momentum': s_days,
              'cds_momentum': s_days, 'volumes_momentum': s_days, 'sma': s5_days, 'volatility': l_days,
              'driftless_volatility': l_days, 'volumes': l_days}

# create data features
start = dt.datetime(2001, 9, 14)
data_wrangler = DataWrangler(data_path='data/', day_ranges=day_ranges, start_date=start, remove_holidays=True)
data_wrangler.save_features()

# normalise features
for ticker in tickers:
    normalise_feature_data(ticker=ticker)

# data exploration
for ticker in tickers:
    data_explorer = DataExploration(ticker=ticker)
    data_explorer.compute_correlations()
    data_explorer.compute_pca()

ne = [64, 128, 256]
tn = [5, 10, 15]
mc = [1, 2, 3, 5]

#  decision tree feature selection
random_forest_save_path = 'data/exploration/dt/rf_cv_scores.csv'
decision_tree_helper = DecisionTreeHelpers(tickers=tickers, num_ests=ne, top_ns=tn, max_counts=mc,
                                           rf_result_path=random_forest_save_path)
if not os.path.exists(random_forest_save_path):
    decision_tree_helper.create_rf_feature_importance_files()
    decision_tree_helper.save_rf_result()

best_ft_choices = {}
ft_importance = {}
rf_results = pd.read_csv(random_forest_save_path, index_col=0)

# create feature data for best feature selection
# not pretty - manual
for ticker in tickers:
    fts = rf_results[f'{ticker}_accuracy'].idxmax()
    best_ft_choices[ticker] = fts
    ft_importance[ticker] = pd.read_csv(f'data/exploration/dt/ft_importance/{ticker}_{fts}.csv', index_col=0)
    num_est = 64 if ticker == 'san' else 128
    decision_tree_helper.get_filtered_feature_importance(ticker=ticker, top_n=10, max_count=2, num_est=num_est)

    # get access to all features through data explorer
    data_explorer = DataExploration(ticker=ticker)
    ft_path = f'data/exploration/dt/ft_importance/{ticker}_top_10_max_2_rf_{num_est}.csv'
    fts = pd.read_csv(ft_path, index_col=0).index.to_list()
    pd.concat([data_explorer.training_data[fts], data_explorer.test_data], axis=1).to_csv(
        f'data/feature_data/{ticker}_best_fts.csv')

# Neural Network Model - illustrative example of loop
batch_size = 64
epochs = 25
train_test_split = 0.80
input_timesteps = 100
validation_size = 0.2
X_val = None
y_val = None

for nn_type in ['lstm']:
    for features_used in ['filter']:
        for epochs in [25]:
            for batch_size in [512]:
                for input_timesteps in [5]:
                    for ticker in tickers:
                        print(nn_type, features_used, epochs, batch_size, input_timesteps, ticker)
                        if features_used == 'filter':
                            data = pd.read_csv(f'data/feature_data/{ticker}_best_fts.csv', index_col=0)
                        elif features_used == 'full':
                            data = pd.read_csv(f'data/feature_data/{ticker}_normalised_features.csv', index_col=0)

                        input_dim = len(data.columns) - 1
                        dl = DataLoader(feature_data=data, test_train_split=train_test_split)

                        # configs can be edited to add layers, adjust parameters, etc.
                        ffn_config = {'loss': 'binary_crossentropy',
                                      'optimiser': 'rmsprop',
                                      1: {'type': 'dense', 'neurons': 16, 'activation': 'relu',
                                          'input_dim': input_dim},
                                      2: {'type': 'dense', 'neurons': 16, 'activation': 'relu'},
                                      3: {'type': 'dense', 'activation': 'sigmoid', 'neurons': 1}}

                        lstm_config = {'loss': 'binary_crossentropy',
                                       'optimiser': 'rmsprop',
                                       1: {'type': 'lstm', 'neurons': 16, 'input_dim': input_dim,
                                           'input_timesteps': input_timesteps + 1, 'return_seq': False},
                                       2: {'type': 'dropout', 'rate': 0.2},
                                       3: {'type': 'dense', 'activation': 'sigmoid', 'neurons': 1}}

                        if nn_type == 'lstm':
                            config = lstm_config
                            X_train, y_train = dl.get_lstm_train_data(input_timesteps + 1)
                            X_test, y_test = dl.get_lstm_test_data(input_timesteps + 1)
                        elif nn_type == 'ffn':
                            config = ffn_config
                            X_train, y_train = dl.get_train_data()
                            X_test, y_test = dl.get_test_data()

                        if validation_size:
                            X_train, X_val, y_train, y_val = dl.get_validation_data(X_train, y_train,
                                                                                    validation_size=validation_size)

                        rn = random.randint(1, 10000)
                        name = f'data_{features_used}_{ticker}_nn_{nn_type}_batchsize_{batch_size}'
                        name += f'_epochs_{epochs}_tts_{train_test_split}_valsize_{validation_size}'
                        if nn_type == 'lstm':
                            name += f'_input_ts_{input_timesteps}'

                        if os.path.exists(f'results/{nn_type}/model_summary/{name}_summary.txt'):
                            continue

                        name += f'_{rn}'
                        m = NeuralNetworkModel(model_specification=config, nn_type=nn_type, name=name)
                        history = m.get_basic_trained_model(x_train=X_train, y_train=y_train,
                                                            epochs=epochs, batch_size=batch_size,
                                                            x_val=X_val, y_val=y_val)

                        if validation_size:
                            plot_network_history(history, name=name, nn_type=nn_type, start_epoch=3)

                        accuracy, avg_predictions = m.get_model_scores(X_test, y_test)
                        print(f'Accuracy: {accuracy}')
                        m.save_summary(nn_type=nn_type, accuracy=accuracy, avg_pred=avg_predictions)
