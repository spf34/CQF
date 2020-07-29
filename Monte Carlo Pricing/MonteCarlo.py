import numpy as np
import pandas as pd
import time


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        return result, te - ts

    return timed


class PathGenerator(object):
    """
    Generates random paths that approximately follow Geometric Brownian Motion
    using one of the below methods:

    Euler-Maruyama: S_{t+1} = S_{t}(1 + (r-d) x dt + sigma x sqrt(dt) x phi)
    Milstein: S_{t+1} = S_{t}(1 + (r-d) x dt + sigma x sqrt(dt) x phi + sigma^2/2 x (phi^2 - 1) x dt)
    Lognormal: S_{t+1} = S_{t} x exp((r-d - sigma^2/2) x dt + sigma x sqrt(dt) x phi)

    phi is a standard normal random variable and dt is time step length

    """

    def __init__(self, s, r, d, t, sigma, randomnes=None):
        self.s = s
        self.r = r
        self.d = d
        self.t = t
        self.sigma = sigma
        self.randomness = randomnes

    def get_randomness(self, num_time_steps, num_paths):
        if self.randomness is None:
            randomness = np.random.normal(0, 1, (num_time_steps, num_paths))
        else:
            randomness = self.randomness
        return randomness

    @timeit
    def log_normal(self, num_paths, num_time_steps):
        dt = self.t / num_time_steps
        randomness = self.get_randomness(num_time_steps, num_paths)
        steps = np.exp(dt * (self.r - self.d - self.sigma ** 2 / 2) + self.sigma * np.sqrt(dt) * randomness)
        return np.vstack([self.s * np.ones(num_paths), self.s * np.cumprod(steps, axis=0)])

    @timeit
    def euler_maruyama(self, num_paths, num_time_steps):
        dt = self.t / num_time_steps
        randomness = self.get_randomness(num_time_steps, num_paths)
        steps = 1 + (self.r - self.d) * dt + self.sigma * np.sqrt(dt) * randomness
        return np.vstack([self.s * np.ones(num_paths), self.s * np.cumprod(steps, axis=0)])

    @timeit
    def milstein(self, num_paths, num_time_steps):
        dt = self.t / num_time_steps
        randomness = self.get_randomness(num_time_steps, num_paths)
        steps = 1 + (self.r - self.d) * dt + self.sigma * np.sqrt(dt) * randomness
        steps += self.sigma ** 2 / 2 * dt * (randomness ** 2 - 1)
        return np.vstack([self.s * np.ones(num_paths), self.s * np.cumprod(steps, axis=0)])

    def generate_paths(self, path_type, num_paths, num_time_steps=1):
        if path_type not in ['log_normal', 'euler_maruyama', 'milstein']:
            raise NotImplementedError(f'Cannot currently generate paths of type: {path_type}')
        method = self.__getattribute__(path_type)
        return method(num_paths=num_paths, num_time_steps=num_time_steps)


class Payoff(object):
    """
    Houses static methods that calculate payoffs
    A MonteCarloPricer evaluates these on paths
    """

    @staticmethod
    def european_call(path, strike):
        return np.maximum(path - strike, 0)

    @staticmethod
    def european_put(path, strike):
        return np.maximum(strike - path, 0)

    @staticmethod
    def european_cash_or_nothing_binary_call(path, strike):
        return 1 * (path > strike)

    @staticmethod
    def european_cash_or_nothing_binary_put(path, strike):
        return 1 * (path < strike)

    @staticmethod
    def european_asset_or_nothing_binary_call(path, strike):
        return path * (path > strike)

    @staticmethod
    def european_asset_or_nothing_binary_put(path, strike):
        return path * (path < strike)


class MonteCarloPricer(object):
    """
    Uses average of realised payoffs to estimate derivatives value:

    1/N sum(P(S_{i)) --> E[P(S)]
    """

    def __init__(self, strike, r, d, t, sample_paths, payoff, return_std_error=False):
        self.r = r
        self.d = d
        self.t = t
        self.paths = sample_paths
        self.num_paths = len(sample_paths)
        self.payoff = payoff
        self.strike = strike
        self.return_std_error = return_std_error

    def compute_european_value(self):
        """

        Returns:
            monte carlo price (float)
            array of running price approximations (np.array)
            standard deviation of (undiscounted) payoff values i.e. standard error (float)

        """
        final_values = self.paths[-1]
        values = self.payoff(final_values, strike=self.strike)
        running_values = values.cumsum() / np.array(range(1, len(final_values) + 1))
        running_values = pd.DataFrame(running_values, index=range(1, len(running_values) + 1))

        mc_value = np.exp(- self.r * self.t) * running_values.iloc[-1, 0]
        running_mc_values = np.exp(- self.r * self.t) * running_values

        if not self.return_std_error:
            std_error = 0
        else:
            std_error = values.std()

        return mc_value, running_mc_values, std_error

    def compute_payoff_standard_deviation(self):
        final_values = self.paths[-1]
        values = self.payoff(final_values, strike=self.strike)
        return values.std()
