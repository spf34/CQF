import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heaviside(x):
    return (np.sign(x) + 1) / 2


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_grad(x):
    return (1 / np.cosh(x)) ** 2


def plot_activation_fcts(input_range):
    plt.figure(figsize=(8, 6))
    plt.plot(input_range, relu(input_range))
    plt.plot(input_range, sigmoid(input_range))
    plt.plot(input_range, np.tanh(input_range))
    plt.legend(['relu', 'sigmoid', 'tanh'])
    plt.title('Activation Functions')
    plt.savefig('activations.png')
    plt.close()


def plot_grads_activation_fcts(input_range):
    plt.figure(figsize=(8, 6))
    plt.plot(input_range, heaviside(input_range))
    plt.plot(input_range, sigmoid_grad(input_range))
    plt.plot(input_range, tanh_grad(input_range))
    plt.legend(['relu', 'sigmoid', 'tanh'])
    plt.title('Activation Function Gradients')
    plt.savefig('activations_grad.png')
    plt.close()


def normalise_feature_data(ticker):
    new_feature_path = f'data/feature_data/{ticker}_normalised_features.csv'
    if not os.path.exists(new_feature_path):
        ticker_path = f'data/feature_data/{ticker}_features.csv'
        data = pd.read_csv(ticker_path, index_col=0)
        train = data.iloc[:, :-1]
        train_norm = (train - train.mean()) / train.std()
        new_data = pd.concat([train_norm, data.iloc[:, -1]], axis=1)
        new_data.to_csv(new_feature_path)


def create_directories():
    for path in ('data/feature_data/', 'data/exploration/',
                 'data/exploration/pca/', 'data/exploration/corr/',
                 'data/exploration/dt/', 'data/exploration/dt/ft_importance',
                 'results/ffn/', 'results/lstm/', 'results/ffn/model_summary', 'results/lstm/model_summary',
                 'results/ffn/figures/', 'results/lstm/figures/',
                 'results/ffn/figures/loss/', 'results/lstm/figures/loss/',
                 'results/lstm/figures/accuracy/', 'results/ffn/figures/accuracy/',
                 'results/lstm/predictions/', 'results/ffn/predictions/'):
        if not os.path.exists(path):
            os.mkdir(path)


def plot_network_history(history, name, nn_type, start_epoch=1):
    """
    Plot accuracy and loss during training

    Adapted from Francois Chollet, 'Deep Learning with Python', 2017.
    """
    history_dict = history.history
    loss_values = history_dict['loss'][start_epoch - 1:]
    val_loss_values = history_dict['val_loss'][start_epoch - 1:]
    acc_values = history_dict['binary_accuracy'][start_epoch - 1:]
    val_acc_values = history_dict['val_binary_accuracy'][start_epoch - 1:]
    epochs = range(start_epoch, start_epoch + len(acc_values))

    plt.figure()
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{nn_type}/figures/loss/{name}.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'results/{nn_type}/figures/accuracy/{name}.png')
    plt.close()


def aggregate_model_accuracy():
    """
    Aggregate all Neural Network summary files
    The files contain accuracy and average prediction information, in addition to the keras summary
    """
    results = {}
    for nn_type in ['ffn', 'lstm']:
        path = f'results/{nn_type}/model_summary/'
        for fname in os.listdir(path):
            try:
                f = open(path + fname, 'r')
                data = f.readlines()[-2:]
                f.close()
                accuracy = float(data[0].split(':')[-1])
                avg_prediction = float(data[1].split(':')[-1])
                results[fname[:fname.find('summary') - 1]] = [accuracy, avg_prediction]
            except:
                print(fname)
    results = pd.DataFrame(results.values(), index=results.keys(), columns=['accuracy', 'avg_prediction'])
    index = results.index.to_list()
    for col, idx in {'ticker': 2, 'nn_type': 4, 'batch_size': 6, 'epochs': 8,
                     'train_test_split': 10, 'window_length': 12, 'val_size': 13}.items():
        results[col] = [x.split('_')[idx] for x in index]
    results.to_csv('results/accuracy_summary.csv')
