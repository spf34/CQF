import os
import numpy as np
import pandas as pd


class DataWrangler(object):
    """
    Converts input csv data downloaded from Bloomberg into features that will act as inputs for our predictive model

    Requires: opens, closes, highs, lows, volumes, price-to-book, gross-total-return, cds closes

    """

    def __init__(self, data_path, day_ranges, prediction_days=1, start_date=None, remove_holidays=False):
        self.feature_flag = False
        self.prediction_days = prediction_days
        self.day_ranges = day_ranges

        data_files = os.listdir(data_path)
        self.attribute_list = ['closes', 'opens', 'highs', 'lows', 'volumes', 'ptb', 'gtrd', 'cds']

        shape = None
        for f in self.attribute_list:
            if not f'{f}.csv' in data_files:
                raise Exception(f'No file of type {f} found in {data_path}')
            data = pd.read_csv(f'data/{f}.csv', index_col=0, parse_dates=True, dayfirst=True)
            if start_date is not None:
                data = data[data.index >= start_date]
            if shape is None:
                shape = data.shape
            else:
                assert data.shape == shape
            self.__setattr__(f, data)

        self.prices = self.closes
        self.attribute_list.append('prices')
        self.tickers = self.prices.columns.to_list()

        if remove_holidays:
            self.remove_holidays()

    def remove_holidays(self):
        """
        Adjust features to remove days with no price moves
        Do not use if we want to have 3 classes - up, down, fixed

        Returns:
            None
        """
        mask = pd.Series(1, index=self.prices.index)
        for t in self.tickers:
            mask = mask & (self.prices.shift(1) != self.prices)[t]

        for name in self.attribute_list:
            attr = self.__getattribute__(name)
            self.__setattr__(name, attr[mask])

    def get_dataframe_from_dict(self, dict, name):
        result = pd.DataFrame()
        for key, data in dict.items():
            data.columns = [f"{t}_{name}_{key}d".lower() for t in self.tickers]
            result = pd.concat([result, data], axis=1)
        return result

    @staticmethod
    def get_sgn_dataframe(data):
        return (abs(data) / data).fillna(0)

    @staticmethod
    def compute_ewma(data, window, persistence=None):
        if persistence is None:
            persistence = 0.9
        weights = np.array([(1 - persistence) * persistence ** t for t in range(window)][::-1])
        weights /= weights.sum()
        ewma_series = pd.DataFrame(weights @ data.iloc[:window], columns=[data.index[window - 1]]).transpose()

        for d in range(window, len(data)):
            next_value = persistence * ewma_series.loc[data.index[d - 1]] + (1 - persistence) * data.iloc[d]
            ewma_series = ewma_series.append(pd.DataFrame(next_value, columns=[data.index[d]]).transpose())

        return ewma_series

    # remove duplication
    def set_returns(self, attr=None):
        attr_str = f'{attr}_returns' if attr is not None else 'returns'
        day_range = self.day_ranges.get(attr_str, [1, 2, 3, 4, 5])
        dict = {}
        if attr is not None:
            data = self.__getattribute__(attr)
        else:
            data = self.prices
        for d in day_range:
            dict[d] = np.log(data / data.shift(d))
        self.__setattr__(attr_str, self.get_dataframe_from_dict(dict, attr_str))

    def set_momentum(self, attr=None):
        attr_str = f'{attr}_momentum' if attr is not None else 'momentum'
        day_range = self.day_ranges.get(attr_str, [1, 2, 3, 4, 5])
        dict = {}
        if attr is not None:
            data = self.__getattribute__(attr)
        else:
            data = self.prices
        for d in day_range:
            dict[d] = data - data.shift(d)
        self.__setattr__(attr_str, self.get_dataframe_from_dict(dict, name=attr_str))

    def set_sma(self):
        day_range = self.day_ranges.get('sma', [5, 10, 15, 20, 25])
        dict = {}
        for d in day_range:
            dict[d] = self.prices.rolling(window=d).mean()
        self.sma = self.get_dataframe_from_dict(dict, name='sma')

    def set_volatility(self):
        day_range = self.day_ranges.get('volatility', [20, 40, 60, 80, 100])
        dict = {}
        for d in day_range:
            returns = self.returns[[f'{t}_returns_1d' for t in self.tickers]]
            dict[d] = returns.rolling(window=d).std()
        self.volatility = self.get_dataframe_from_dict(dict, 'volatility')

    def set_driftless_volatility(self):
        """
        Set driftless volatility, as defined in Yang, Zhang (2000)

        Returns:
            None

        """
        day_range = self.day_ranges.get('volatility', [20, 40, 60, 80, 100])
        dict = {}

        o = np.log(self.opens / self.closes.shift(1))
        u = np.log(self.highs / self.opens)
        d = np.log(self.lows / self.opens)
        c = np.log(self.closes / self.opens)

        for day in day_range:
            k = 0.34 / (1.34 + (day + 1) / (day - 1))
            v_o = o.rolling(window=day).std() ** 2
            v_c = c.rolling(window=day).std() ** 2
            v_rs = (u * (u - c) + d * (d - c)).rolling(window=day).mean()
            v = v_o + k * v_c + (1 - k) * v_rs
            dict[day] = np.sqrt(256 * v)
        self.driftless_volatility = self.get_dataframe_from_dict(dict, 'driftless_volatility')

    def set_sgn_returns(self):
        day_range = self.day_ranges.get('sgn_returns', [1, 2, 3, 4, 5])
        dict = {}
        for d in day_range:
            dict[d] = self.get_sgn_dataframe(self.returns[[f'{t}_returns_{d}d' for t in self.tickers]])
        self.sgn_returns = self.get_dataframe_from_dict(dict, 'sgn_returns')

    def set_median_volumes(self):
        day_range = self.day_ranges.get('volumes', [20, 40, 60, 80, 100])
        dict = {}
        for d in day_range:
            dict[d] = self.volumes.rolling(window=d).median()
        self.median_volumes = self.get_dataframe_from_dict(dict, 'median_volumes')

    def set_dvds(self):
        dict = {}
        gtr = (self.gtrd - self.gtrd.iloc[0]) / 100
        cumulative_return = self.prices / self.prices.iloc[0] - 1
        dict[1] = gtr - cumulative_return
        self.dvd_proxy = self.get_dataframe_from_dict(dict, 'dvd_proxy')

    def set_stochastic_oscillator(self):
        """
        Stochastic K
        Technical indicator useful for determining when a security has been overbought or oversold

        Returns:
            None
        """
        day_range = self.day_ranges.get('stochastic_oscillator', [5, 14])
        dict = {}
        for d in day_range:
            lows = self.lows.rolling(d).min()
            highs = self.highs.rolling(d).max()
            dict[d] = (self.prices - lows) / (highs - lows)
        self.stochastic_oscillator = self.get_dataframe_from_dict(dict, 'stochastic_oscillator')

    def set_rsi(self):
        """
        RSI - Relative Strength Index
        Technical indicator showing whether a security may be overbough or oversold
        rsi = rs/(1 + rs)
        rs = sma(up_moves)/sma(down_moves)

        Returns:
            None
        """
        day_range = self.day_ranges.get('rsi', [14])
        dict = {}
        mtm = self.momentum[[f'{t}_momentum_1d' for t in self.tickers]]
        up = np.maximum(mtm, 0)
        down = -np.minimum(mtm, 0)
        for d in day_range:
            count_ups = (up > 0).rolling(window=d).sum()
            count_downs = (down > 0).rolling(window=d).sum()
            up_avg = up.rolling(window=d).sum() / count_ups
            down_avg = down.rolling(window=d).sum() / count_downs
            dict[d] = up_avg / (up_avg + down_avg)
        self.rsi = self.get_dataframe_from_dict(dict, 'rsi')

    def set_macd(self):
        """
        MACD - Moving Average Convergence/Divergence
        Technical indicator showing momentum trends: ewma_12 - ewma_26

        Returns:
            None
        """
        day_range = self.day_ranges.get('macd', [[12, 26]])
        dict = {}
        for short, long in day_range:
            ewma_short = DataWrangler.compute_ewma(self.prices, window=short, persistence=1 - 1 / (2 * short + 1))
            ewma_long = DataWrangler.compute_ewma(self.prices, window=long, persistence=1 - 1 / (2 * long + 1))
            dict[f'{short}_{long}'] = ewma_short - ewma_long
        self.macd = self.get_dataframe_from_dict(dict, 'macd')

    def set_cci(self):
        """
        CCI - Commodity Channel Index
        Technical indicator for identifying cyclical trends

        Returns:
            None
        """
        day_range = self.day_ranges.get('cci', [20])
        dict = {}
        for d in day_range:
            typical_price = (self.closes + self.highs + self.lows)
            typical_price_sma = typical_price.rolling(window=d).mean()
            typical_price_md = abs(typical_price - typical_price_sma).rolling(window=d).mean()
            dict[d] = (typical_price - typical_price_sma) / typical_price_md
        self.cci = self.get_dataframe_from_dict(dict, 'cci')

    def set_atr(self):
        """
        ATR - Average True Range
        A technical indicator measuring volatility

        Returns:
            None
        """
        day_range = self.day_ranges.get('atr', [14])
        dict = {}
        for d in day_range:
            true_range = np.maximum(self.highs - self.lows, self.highs - self.closes.shift(1))
            true_range = np.maximum(true_range, self.closes.shift(1) - self.lows).dropna()
            dict[d] = DataWrangler.compute_ewma(true_range, window=d, persistence=1 - 1 / d)
        self.atr = self.get_dataframe_from_dict(dict, 'atr')

    def set_ad(self):
        """
        AD - Accumulation / Distribution
        Technical indicator representing volume weighted investor sentiment

        Returns:
            None
        """
        day_range = self.day_ranges.get('ad', [1])
        dict = {}
        for d in day_range:
            clv = (2 * self.closes - self.highs - self.lows) / (self.highs - self.lows)
            dict[d] = (self.volumes * clv).cumsum() / self.volumes.median()
        self.ad = self.get_dataframe_from_dict(dict, 'ad')

    def create_feature_data(self):
        self.set_returns()
        self.set_momentum()
        self.set_sma()
        self.set_volatility()
        self.set_driftless_volatility()
        self.set_sgn_returns()

        # cds and volumes
        self.set_returns(attr='cds')
        self.set_momentum(attr='cds')
        self.set_momentum(attr='volumes')
        self.set_median_volumes()

        # fundamentals
        self.set_dvds()

        # technical indicators
        self.set_stochastic_oscillator()
        self.set_rsi()
        self.set_macd()
        self.set_cci()
        self.set_atr()
        self.set_ad()

        results = self.sgn_returns[[f'{t}_sgn_returns_1d' for t in self.tickers]].shift(-self.prediction_days)
        results.columns = [f'{c.replace("sgn_returns", "results")}' for c in results.columns]
        self.results = results
        self.feature_flag = True

    def save_features_by_ticker(self, features):
        features_columns = features.columns
        for t in self.tickers:
            columns = [c for c in features_columns if t in c]
            data = features[columns].dropna()
            data.to_csv(f'data/feature_data/{t}_features.csv')

    def save_features(self):
        if os.path.exists('data/feature_data/features.csv'):
            return

        if not self.feature_flag:
            self.create_feature_data()

        features = pd.DataFrame()
        for feature in ['returns', 'momentum', 'sma', 'volatility', 'driftless_volatility', 'sgn_returns',
                        'cds_returns', 'cds_momentum', 'volumes_momentum', 'median_volumes', 'dvd_proxy',
                        'stochastic_oscillator', 'rsi', 'macd', 'cci', 'atr', 'ad', 'results']:
            features = pd.concat([features, self.__getattribute__(feature)], axis=1)
        features = features.dropna(how='any', axis=0)
        features.to_csv('data/feature_data/features.csv')

        self.save_features_by_ticker(features)
