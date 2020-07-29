import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eig

plt.style.use('ggplot')


# get_data to be run before main
# get_data should take 1-2 minutes
# main may take up to 20 minutes


def get_sign_dataframe(data):
    return (abs(data) / data).fillna(0)


def split_ticker(t):
    return f"{t.replace(' ', '_').replace('/', '').split('_')[0]}"


def get_prices(price_path, tickers=None):
    prices = pd.read_csv(price_path, index_col=0)
    prices.columns = [c.lower().split()[0].replace('/', '') for c in prices.columns]
    if tickers is not None:
        prices = prices[list(tickers)]

    # remove duplicates as proxies for holidays
    mask = pd.Series(0, index=prices.index)
    for t in tickers:
        mask = mask | (prices.shift() != prices)[t]
    return prices[mask]


def compute_ewma(data, window, persistence=None):
    if persistence is None:
        persistence = 0.9
    weights = np.array([persistence * (1 - persistence) ** t for t in range(window)][::-1])
    weights /= weights.sum()
    ewma_series = pd.DataFrame(weights @ data.iloc[:window], columns=[data.index[window - 1]]).transpose()

    for d in range(window, len(data)):
        next_value = persistence * ewma_series.loc[data.index[d - 1]] + (1 - persistence) * data.iloc[d]
        ewma_series = ewma_series.append(pd.DataFrame(next_value, columns=[data.index[d]]).transpose())

    return ewma_series


def get_dataframe_from_dict(dict, name):
    result = pd.DataFrame()
    for key, data in dict.items():
        data.columns = [f"{split_ticker(t)}_{name}_{key}d".lower() for t in data.columns]
        result = pd.concat([result, data], axis=1)
    return result


def features_from_prices(price_path, tickers=None):
    prices = get_prices(price_path, tickers=tickers)

    returns_dict = {}
    momentum_dict = {}
    sma_dict = {}
    ewma_dict = {}
    std_dict = {}

    # returns and momentum
    for d in range(1, 6):
        returns_dict[d] = np.log(prices / prices.shift(d))
        momentum_dict[d] = prices - prices.shift(d)
    momentum = get_dataframe_from_dict(momentum_dict, name='momentum')
    returns = get_dataframe_from_dict(returns_dict, name='returns')

    # moving averages
    for d in range(5, 30, 5):
        sma_dict[d] = prices.rolling(window=d).mean()
        ewma_dict[d] = compute_ewma(prices, window=d)
    sma = get_dataframe_from_dict(sma_dict, name='sma')
    ewma = get_dataframe_from_dict(ewma_dict, name='ewma')

    # volatility
    rtns = returns_dict[1]
    for d in range(20, 120, 20):
        std_dict[d] = rtns.rolling(window=d).std()
    std = get_dataframe_from_dict(std_dict, 'std')

    signed_returns = get_sign_dataframe(rtns)
    signed_returns.columns = [f"{c.split('_')[0]}_returns_sr" for c in signed_returns.columns]

    return pd.concat([momentum, returns, sma, ewma, std, signed_returns], axis=1)


def add_additional_features(features, tickers=None):
    highs = pd.read_csv(f'data/highs.csv', index_col=0)
    lows = pd.read_csv(f'data/lows.csv', index_col=0)
    closes = pd.read_csv(f'data/prices.csv', index_col=0)
    opens = pd.read_csv(f'data/opens.csv', index_col=0)
    dfs = [highs, lows, opens, closes]
    if tickers is not None:
        for idx in range(len(dfs)):
            dfs[idx] = dfs[idx][list(tickers)]
    highs, lows, opens, closes = dfs
    intraday_momentum = get_sign_dataframe(closes - opens)
    intraday_momentum.columns = [f'{c}_momentum_im' for c in intraday_momentum.columns]
    delta_highs = abs(closes - highs)
    delta_lows = abs(closes - lows)
    hl_momentum = 2 * (delta_highs < delta_lows) - 1
    hl_momentum.columns = [f'{c}_momentum_hlm' for c in hl_momentum.columns]
    return pd.concat([features, intraday_momentum, hl_momentum], axis=1)


def save_features_by_ticker(features, tickers):
    features_columns = features.columns
    tickers = [t for t in set([t.split('_')[0] for t in features_columns]) if t in tickers]
    for t in tickers:
        columns = [c for c in features_columns if t in c]
        data = features[columns].dropna()
        data.to_csv(f"data/feature_data/{t}_features.csv")


def compute_correlations(data, ticker):
    data.corr().to_csv(f"data/results/corr/_{ticker}_overall_corr.csv")
    for ft in feature_types:
        columns = [c for c in data.columns if ft in c]
        temp = data[columns]
        temp.columns = [c.split('_')[-1] for c in temp.columns]
        corr = temp.corr()
        corr.to_csv(f"data/results/corr/{ticker}_{ft}.csv")
        plt.figure()
        sns.heatmap(corr)
        plt.title(f"{ticker} {ft}")
        plt.savefig(f"data/results/corr/figures/{ticker}_{ft}.png")
        plt.close()


def perform_pca(data):
    corr = data.corr()
    values, vectors = eig(corr)
    return values / sum(values)


def create_pca_plots(features):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    for t in tickers:
        pca = perform_pca(features[t])
        plt.plot(range(1, len(pca) + 1), 100 * perform_pca(features[t]), 'x')
    plt.legend(tickers)
    plt.title('Feature PCA - % Variance by Eigenvector')
    plt.ylabel('% Variance')

    plt.subplot(2, 1, 2)
    for t in tickers:
        pca = perform_pca(features[t])
        plt.plot(range(1, len(pca) + 1), 100 * pca.cumsum(), 'x')
    plt.legend(tickers)
    plt.xlabel('Eigenvector Number')
    plt.ylabel('Cumulative % Variance')
    plt.savefig('data/results/pca/contributions_to_variance.png')
    plt.close()


def add_result_columns(features):
    for ticker in tickers:
        result_column = f'{ticker}_result'
        sr_column = f'{ticker}_returns_sr'
        feature = features[ticker]

        # restrict to dates with a price move
        dates = feature[feature[sr_column] != 0].index.tolist()
        feature = feature[feature.index.isin(dates)]

        # add +/- next day price direction for comparison
        feature.loc[:, result_column] = feature[sr_column].shift(-1)
        feature.loc[:, f'{ticker}_result_zero_one'] = (feature[result_column] + 1) / 2
        features[ticker] = feature

    return features


def plot_rebased_prices(rebased_prices):
    plt.figure(figsize=(8, 6))
    for ticker in tickers:
        plt.plot(rebased_prices[ticker].values)
    plt.title('Rebased Prices')
    plt.xlabel('Date')
    plt.legend(tickers)
    plt.savefig('data/results/analysis/prices.png')
    plt.close()


def plot_return_histogram(prices):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    bins = [0.01 * x for x in range(-10, 11)]
    plt.figure(figsize=(8, 6))
    plt.hist(log_returns.values, bins=bins)
    plt.title('Returns Histogram')
    plt.xlabel('Date')
    plt.legend(tickers)
    plt.savefig('data/results/analysis/returns_hist.png')
    plt.close()


def count_up_down_moves(tickers):
    results = {}
    for ticker in tickers:
        features = pd.read_csv(f'data/feature_data/{ticker}_final_data.csv', index_col=0)
        column_name = f'{ticker}_result_zero_one'
        count = len(features)
        up = features[column_name].sum()
        down = count - up
        results[ticker] = {'count': count, 'up': up, 'down': down, 'up_ratio': up / count,
                           'down_ratio': down / count}
    results = pd.DataFrame(results)
    results.to_csv('data/results/analysis/up_down.csv')


tickers = ('pl1', 'hg1')
feature_types = ('momentum', 'returns', 'sma', 'ewma', 'std')

if __name__ == '__main__':
    # create directories
    for path in ['data/', 'data/results/', 'data/feature_data/', 'data/results/pca/', 'data/results/corr/',
                 'data/results/corr/figures', 'data/results/analysis/', 'data/results/forward_selection',
                 'data/results/exhaustive_selection', 'data/results/grid_cv/', 'data/results/bt/']:
        if not os.path.exists(path):
            os.mkdir(path)

    price_path = f'data/prices.csv'
    features_path = f'data/feature_data/features.csv'
    if not os.path.exists(features_path):
        features = features_from_prices(price_path, tickers=tickers)
        features = add_additional_features(features, tickers=tickers)
        features.to_csv(features_path)
    else:
        features = pd.read_csv(features_path, index_col=0)

    if any([not os.path.exists(f"data/feature_data/{t}_features.csv") for t in tickers]):
        save_features_by_ticker(features, tickers=tickers)

    for t in tickers:
        data = pd.read_csv(f"data/feature_data/{t}_features.csv", index_col=0)
        compute_correlations(data, t)

    feature_path = f'data/feature_data/'
    features = {}
    for t in tickers:
        features[t] = pd.read_csv(f'{feature_path}{t}_features.csv', index_col=0)
    if not os.path.exists(f'data/results/pca/contributions_to_variance.png'):
        create_pca_plots(features)

    features = add_result_columns(features)
    for ticker in tickers:
        features[ticker].iloc[:-1].to_csv(f'data/feature_data/{ticker}_final_data.csv')

    prices = pd.read_csv('data/prices.csv', index_col=0)
    rebased_prices = 100 * prices.divide(prices.iloc[0], axis=1)[list(tickers)]

    plot_rebased_prices(rebased_prices)
    plot_return_histogram(rebased_prices)

    count_up_down_moves(tickers)
