from Portfolio import Covariance
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class ValueAtRisk(object):
    """
    Basic Value at Risk calculator using assumption of normally distributed returns
    Volatility input should be daily. All assets are assumed to be denominated in the same currency
    """

    def __init__(self, volatility, expected_return=0, notional=0):
        self.vol = volatility
        self.expected_return = expected_return
        self.notional = notional

    def get_simple_percentile_VaR(self, days=10, percentile=99):
        period_vol = np.sqrt(days) * self.vol
        normal_factor = norm.ppf(1 - percentile / 100)
        return self.expected_return + period_vol * normal_factor

    def get_notional_simple_percentile_VaR(self, days=10, percentile=99):
        if not self.notional:
            print('No Notional value has been specified')
            return
        # round result to 2dp since this is a cash value
        return round(self.notional * self.get_simple_percentile_VaR(days=days, percentile=percentile), 2)

    def get_expected_shortfall(self, days=10, percentile=99):
        c = percentile / 100
        return self.expected_return - self.vol * np.sqrt(days) / ((1 - c) * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * (norm.ppf(1 - c)) ** 2)


class PortfolioValueAtRisk(ValueAtRisk):
    """
    Basic Value at Risk calculator using assumption of jointly normally distributed returns
    """

    def __init__(self, notionals, expected_returns=None, cov=None, corr=None, std=None):
        covariance_object = Covariance(cov=cov, corr=corr, std=std)
        self.cov = covariance_object.cov
        self.std = covariance_object.std

        if len(notionals) != len(self.std):
            raise BaseException('Number of assets do not agree across notionals and covariance')
        notional = sum(notionals)

        self.weights = np.array([n / notional for n in notionals])
        volatility = np.sqrt(self.weights @ (self.cov @ self.weights))

        if expected_returns is not None:
            mu = self.weights @ expected_returns
        else:
            mu = 0
        super(PortfolioValueAtRisk, self).__init__(volatility=volatility, expected_return=mu, notional=notional)


class VarBacktest(object):
    def __init__(self, price_data, volatility_estimation_window=21, testing_period=10):
        price_data.index = pd.to_datetime(price_data.index)
        self.prices = price_data
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.window = volatility_estimation_window
        self.testing_period = testing_period

    def compute_volatility_estimate(self):
        raise NotImplementedError

    def compute_var_estimate(self, level=0.99):
        normal_factor = norm.ppf(1 - level)
        volatility_estimate = self.compute_volatility_estimate()
        return np.sqrt(self.testing_period) * normal_factor * volatility_estimate

    def compute_testing_period_returns(self):
        return np.log(self.prices.shift(-self.testing_period) / self.prices).dropna()[self.window:]

    def get_breaches(self, level=0.99):
        var = self.compute_var_estimate(level=level)
        test_returns = self.compute_testing_period_returns()
        var = var[var.index.isin(test_returns.index)]
        return 1 * (var > test_returns)

    def get_consecutive_breaches(self, level=0.99):
        breaches = self.get_breaches()
        return breaches * breaches.shift(1)

    def get_headline_figures(self):
        """

        Return:
            breaches (int)
            proportion of breaches (float)
            consecutive breaches (int)
            conditional probability of consecutive breach (float)
        """

        breaches = self.get_breaches()
        num_breaches = breaches.sum()[0]
        pct_breaches = num_breaches / len(breaches)
        consecutive_breaches = self.get_consecutive_breaches()
        num_cb = consecutive_breaches.sum()[0]
        conditional_prob_cb = num_cb / num_breaches
        return num_breaches, pct_breaches, num_cb, conditional_prob_cb

    def plot_var_versus_realised_return(self, level=0.99):
        rtns = self.compute_testing_period_returns()

        breaches = self.get_breaches(level=level)
        breach_dates = breaches[breaches > 0].dropna().index
        breach_returns = rtns[rtns.index.isin(breach_dates)]
        consecutive_breaches = self.get_consecutive_breaches(level=level)
        consecutive_dates = consecutive_breaches[consecutive_breaches > 0].dropna().index
        consecutive_breach_returns = rtns[rtns.index.isin(consecutive_dates)]

        plt.plot(self.compute_var_estimate(level=level))
        plt.plot(self.compute_testing_period_returns())
        plt.scatter(breach_dates, breach_returns, marker='_', color='orange', s=200)
        plt.scatter(consecutive_dates, consecutive_breach_returns, marker='_', color='darkred', s=200)
        plt.legend(
            [f'{round(100 * level, 2)}% VaR Estimate', '10D Realised Returns', 'Breaches', 'Consecutive Breaches'])
        plt.ylabel('10D Return')
        plt.title(f'10D {round(100 * level, 2)}% VaR vs Realised Returns')


class VarBacktestRollingVolatility(VarBacktest):
    def __init__(self, price_data, volatility_estimation_window=21, testing_period=10):
        super(VarBacktestRollingVolatility, self).__init__(price_data=price_data,
                                                           volatility_estimation_window=volatility_estimation_window,
                                                           testing_period=testing_period)

    def compute_volatility_estimate(self):
        return self.returns.rolling(window=self.window).std()

    def plot_var_versus_realised_return(self, level=0.99):
        super(VarBacktestRollingVolatility, self).plot_var_versus_realised_return(level=level)
        plt.title(f'10D {round(100 * level, 2)}% VaR vs Realised Returns - Rolling Volatility')


class VarBacktestEWMAVolatility(VarBacktest):
    def __init__(self, price_data, volatility_estimation_window=21, testing_period=10,
                 lambda_coefficient=0.72, renorm=True):
        super(VarBacktestEWMAVolatility, self).__init__(price_data=price_data,
                                                        volatility_estimation_window=volatility_estimation_window,
                                                        testing_period=testing_period)
        self.persistence = lambda_coefficient
        self.renormalise_weights = renorm

        weights = [np.power(self.persistence, t) * (1 - self.persistence) for t in range(self.window)][::-1]
        if self.renormalise_weights:
            weights /= sum(weights)
        self.weights = weights

    def compute_volatility_estimate(self, start_method=2):
        """

        start_method (int): integer to determine method for initialising the volatility series
                            1 - use sample volatility from full series
                            2 - use EWMA weights on window prior to first date
        Return:
        """

        if start_method == 1:
            square_rtns = self.returns[:self.window] ** 2
            initial = np.sqrt(self.weights @ square_rtns)[0]
        elif start_method == 2:
            initial = self.returns.std()[0]
        else:
            raise BaseException(f'Start method: {start_method} not yet implemented')

        idx = self.compute_testing_period_returns().index
        square_rtns = self.returns[self.returns.index.isin(idx)] ** 2

        ewma_vol = pd.Series(index=idx)
        ewma_vol.iloc[0] = initial
        for i in range(1, len(ewma_vol)):
            new_variance = self.persistence * ewma_vol.iloc[i - 1] ** 2 + (1 - self.persistence) * square_rtns.iloc[i]
            ewma_vol.iloc[i] = np.sqrt(new_variance)[0]
        return pd.DataFrame(ewma_vol, columns=['SP500'])

    def plot_var_versus_realised_return(self, level=0.99):
        super(VarBacktestEWMAVolatility, self).plot_var_versus_realised_return(level=level)
        plt.title(f'10D {round(100 * level, 2)}% VaR vs Realised Returns - EWMA Volatility')
