from MonteCarlo import MonteCarloPricer, PathGenerator

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class PricingHelpers(object):

    @staticmethod
    def compare_european_mc(k, r, d, t, paths, payoff, analytical_value, return_std_error=False):
        """

        Basic method used for comparing Monte Carlo estimate to analytical price

        Returns:
            mc_value (float):
            mc_running_values
            abs_error (float):
            rel_error (float):
            std_error (float):

        """
        pricer = MonteCarloPricer(strike=k, r=r, d=d, t=t, sample_paths=paths, payoff=payoff,
                                  return_std_error=return_std_error)

        mc_value, mc_running_values, std_error = pricer.compute_european_value()
        abs_error = abs(mc_value - analytical_value)
        rel_error = abs(mc_value / analytical_value - 1)
        print(f'True Value: {analytical_value}, MC Call Value: {mc_value}')
        print(f'Absolute Difference: {abs_error}, Relative Difference: {round(100 * rel_error, 2)}%')

        return mc_value, mc_running_values, abs_error, rel_error, std_error

    @staticmethod
    def repeated_pricing(s, k, r, d, t, sigma, payoff, num_paths, num_timesteps, repeats=100, path_type='log_normal',
                         column_name='mc_values', return_running_values=False):
        path_generator = PathGenerator(s, r, d, t, sigma)
        result = []
        times = []

        if return_running_values:
            running = pd.DataFrame()

        for rp in range(repeats):
            paths, run_time = path_generator.generate_paths(path_type=path_type, num_paths=num_paths,
                                                            num_time_steps=num_timesteps)
            pricer = MonteCarloPricer(strike=k, r=r, d=d, t=t, sample_paths=paths, payoff=payoff)
            mc_value, mc_running_values, runtime = pricer.compute_european_value()
            result.append(mc_value)
            times.append(run_time)

            if return_running_values:
                running = running.transpose().append(mc_running_values.transpose()).transpose()

        result = pd.DataFrame({column_name: result, f'{column_name}_time': times})
        if return_running_values:
            running.columns = range(1, repeats + 1)
            return result, running
        return result

    @staticmethod
    def add_error_columns_to_results(results, analytical_price, price_column):
        results['abs_vs_exact'] = results[price_column] - analytical_price
        results['rel_vs_exact'] = results[price_column] / analytical_price - 1
        results['rel_vs_exact[%]'] = results.rel_vs_exact.apply(lambda x: f'{round(x * 100, 2)}%')
        return results

    @staticmethod
    def get_grid_priced_results(s, k, r, d, t, sigma, grid, path_type, payoff):

        """
        Args:
            s (float): stock price
            k (float): strike
            r (float): interest rate
            d (float): dividend yield
            t (float): time to maturity [years]

            sigma (float): annualised volatility
            grid (np.array): grid of path descriptions --> (num_path, num_time_steps)
            path_type (str): scheme used to simulate paths from ['log_normal', 'euler_murayama', 'milstein']
            payoff (MonteCarlo.Payoff): payoff method

        Returns:
            results (pd.DataFrame): mc prices, as well as time to compute, std error and accuracy figures

        """
        path_generator = PathGenerator(s, r, d, t, sigma)
        results = {}
        times = {}
        std_errors = {}

        for num_paths, num_time_steps in grid:
            idx = (num_paths, num_time_steps)
            paths, run_time = path_generator.generate_paths(path_type, num_paths=num_paths,
                                                            num_time_steps=num_time_steps)
            pricer = MonteCarloPricer(strike=k, r=r, d=d, t=t, sample_paths=paths, payoff=payoff, return_std_error=True)
            mc_value, _, std_error = pricer.compute_european_value()

            results[idx] = mc_value
            times[idx] = run_time
            std_errors[idx] = std_error

        results = pd.DataFrame(list(results.values()), index=list(results.keys()), columns=['mc_price'])
        results['time'] = times.values()
        results['num_paths'] = [x[0] for x in results.index]
        results['num_time_steps'] = [x[1] for x in results.index]
        results['accuracy'] = 1 / (np.sqrt(results.num_paths) * results.num_time_steps)
        results['path_type'] = path_type

        return results

    @staticmethod
    def price_by_grid(s, k, r, d, t, sigma, payoff, path_type='log_normal', analytical_price=0, grid=None):

        """

        Wrapper for get_grid_priced_results where grid is supplied

        """
        if grid is None:
            print('No Grid provided')
            return

        results = PricingHelpers.get_grid_priced_results(s=s, k=k, r=r, d=d, t=t, sigma=sigma, grid=grid,
                                                         path_type=path_type, payoff=payoff)
        if analytical_price:
            results = PricingHelpers.add_error_columns_to_results(results, analytical_price, price_column='mc_price')
        return results

    @staticmethod
    def price_by_accuracy(s, k, r, d, t, sigma, payoff, path_type='log_normal', analytical_price=0, accuracies=None):
        """

        Wrapper for get_grid_priced_results where the grid is determined by input accuracies
        For accuracy epsilon, we use 1/epsilon ** 2 paths and 1/epsilon time steps

        """
        if accuracies is None:
            accuracies = [0.02, 0.01, 0.005, 0.025]
        num_paths = [int(1 / x ** 2) for x in accuracies]
        num_time_steps = [int(1 / x) for x in accuracies]
        grid = zip(num_paths, num_time_steps)

        results = PricingHelpers.get_grid_priced_results(s=s, k=k, r=r, d=d, t=t, sigma=sigma, grid=grid,
                                                         path_type=path_type, payoff=payoff)
        if analytical_price:
            results = PricingHelpers.add_error_columns_to_results(results, analytical_price, price_column='mc_price')
        return results

    @staticmethod
    def payoff_varying_parameters(parameters, payoff):
        """

        Args:
            parameters (pd.DataFrame): dataframe of parameter values to use as payoff arguments
                                       columns must include 's', 'k', 'r', 'd', 't', 'sigma'
            payoff (AnalyticalHelpers.method): method to compute analytical value

        Returns:

        """
        results = parameters.copy()
        result_column = []
        for idx, row in parameters.iterrows():
            result_column.append(payoff(s=row.s, k=row.k, r=row.r, d=row.d, t=row.t, sigma=row.sigma))
        results['value'] = result_column
        return results

    @staticmethod
    def plot_by_spot_and_time(s, k, r, d, sigma, payoff, times_to_maturity=None):
        """

        plot analytical payoff for a collection of times to maturity

        """
        parameter_to_vary = 's'
        parameter_values = [k / 2 + 0.1 * x for x in range(1, 10 * k + 1)]
        if times_to_maturity is None:
            times_to_maturity = [1.5, 1.25, 1, 0.75, 0.5, 0.25]
        for t in times_to_maturity:
            parameters = pd.DataFrame({'s': s, 'k': k, 'r': r, 'd': d, 't': t, 'sigma': sigma},
                                      index=list(range(len(parameter_values))))
            parameters[parameter_to_vary] = parameter_values
            results = PricingHelpers.payoff_varying_parameters(parameters=parameters, payoff=payoff)
            plt.plot(results[[parameter_to_vary, 'value']].set_index(parameter_to_vary))
        plt.legend(times_to_maturity)


class AnalyticalHelpers(object):
    """
    Static methods that encasulate analytical formulas for
    pricing derivatives, computing greeks and making use of put-call parity
    """

    @staticmethod
    def get_d1(s, k, r, d, t, sigma):
        return (np.log(s / k) + (r - d + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    @staticmethod
    def get_d2(s, k, r, d, t, sigma):
        return (np.log(s / k) + (r - d - sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))

    # option pricing formulas
    @staticmethod
    def analytical_european_call(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return s * np.exp(- d * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def analytical_european_put(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    @staticmethod
    def analytical_european_asset_or_nothing_binary_call(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return s * np.exp(- d * t) * norm.cdf(d1)

    @staticmethod
    def analytical_european_asset_or_nothing_binary_put(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return s * np.exp(- d * t) * norm.cdf(-d1)

    @staticmethod
    def analytical_european_cash_or_nothing_binary_call(s, k, r, d, t, sigma, payoff=1):
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return payoff * np.exp(- r * t) * norm.cdf(d2)

    @staticmethod
    def analytical_european_cash_or_nothing_binary_put(s, k, r, d, t, sigma, payoff=1):
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return payoff * np.exp(- r * t) * norm.cdf(-d2)

    # greeks

    # deltas
    @staticmethod
    def european_call_delta(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return np.exp(-d * t) * norm.cdf(d1)

    @staticmethod
    def european_put_delta(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return -np.exp(-d * t) * norm.cdf(-d1)

    @staticmethod
    def cash_or_nothing_european_call_delta(s, k, r, d, t, sigma):
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return np.exp(-r * t) * norm.pdf(d2) / (sigma * np.sqrt(t) * s)

    @staticmethod
    def cash_or_nothing_european_put_delta(s, k, r, d, t, sigma):
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return -np.exp(-r * t) * norm.pdf(d2) / (sigma * np.sqrt(t) * s)

    @staticmethod
    def asset_or_nothing_european_call_delta(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return np.exp(-d * t) * (norm.cdf(d1) + norm.pdf(d1) / (sigma * np.sqrt(t)))

    @staticmethod
    def asset_or_nothing_european_put_delta(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return np.exp(-d * t) * (norm.cdf(-d1) - norm.pdf(-d1) / (sigma * np.sqrt(t)))

    # gammas
    @staticmethod
    def vanilla_european_gamma(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        return np.exp(-d * t) * norm.pdf(d1) / (sigma * np.sqrt(t) * s)

    @staticmethod
    def cash_or_nothing_european_call_gamma(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return -np.exp(-r * t) * d1 * norm.cdf(d2) / (s ** 2 * sigma ** 2 * t)

    @staticmethod
    def cash_or_nothing_european_put_gamma(s, k, r, d, t, sigma):
        return -AnalyticalHelpers.cash_or_nothing_european_call_gamma(s, k, r, d, t, sigma)

    @staticmethod
    def asset_or_nothing_european_call_gamma(s, k, r, d, t, sigma):
        d1 = AnalyticalHelpers.get_d1(s, k, r, d, t, sigma)
        d2 = AnalyticalHelpers.get_d2(s, k, r, d, t, sigma)
        return -np.exp(-d * t) * norm.pdf(d1) * d2 / (s * sigma ** 2 * t)

    @staticmethod
    def asset_or_nothing_european_put_gamma(s, k, r, d, t, sigma):
        return -AnalyticalHelpers.asset_or_nothing_european_call_gamma(s, k, r, d, t, sigma)

    # put-call parity relationships
    @staticmethod
    def call_to_put(c, s, k, r, d, t):
        return c - s * np.exp(-d * t) + k * np.exp(-r * t)

    @staticmethod
    def put_to_call(p, s, k, r, d, t):
        return p + s * np.exp(-d * t) - k * np.exp(-r * t)

    @staticmethod
    def binary_asset_or_nothing_call_to_put(c, s, r, d, t):
        return s * np.exp(-d * t) - c

    @staticmethod
    def binary_asset_or_nothing_put_to_call(p, s, r, d, t):
        return s * np.exp(-d * t) - p

    @staticmethod
    def binary_cash_or_nothing_call_to_put(c, r, t, payoff=1):
        return payoff * np.exp(-r * t) - c

    @staticmethod
    def binary_cash_or_nothing_put_to_call(p, r, t, payoff=1):
        return payoff * np.exp(-r * t) - p
