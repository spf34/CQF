import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class Covariance(object):
    """
    Covariance object that can convert between correlation and volatility (std) and covariance

    """

    def __init__(self, cov=None, corr=None, std=None):
        self._cov = cov
        self._corr = corr
        self._std = std

        if cov is not None:
            if not Covariance.check_symmetric(cov):
                raise BaseException('Covariance matrix should be symmetric')
            cov = Covariance.convert_to_array(cov)
            self._cov = cov
        elif (corr is not None and std is not None):
            if not Covariance.check_symmetric(corr):
                raise BaseException('Correlation matrix should be symmetric')
            corr = Covariance.convert_to_array(corr)
            self._corr = corr
            self._cov = Covariance.corr_to_cov(corr, std)
        else:
            raise BaseException('Must provide either "cov" or ("corr" and "std")')

    @property
    def cov(self):
        if self._cov is None:
            self._cov = self.corr_to_cov(self.corr, self.std)
        return self._cov

    @property
    def corr(self):
        if self._corr is None:
            self._corr, self._std = self.cov_to_corr_std(self.cov)
        return self._corr

    @property
    def std(self):
        if self._std is None:
            self._corr, self._std = self.cov_to_corr_std(self.cov)
        return self._std

    @staticmethod
    def check_symmetric(matrix):
        if isinstance(matrix, list):
            matrix_ = np.array(matrix)
        else:
            matrix_ = matrix
        transpose = matrix_.transpose()
        return np.all(matrix_ == transpose)

    @staticmethod
    def convert_to_array(matrix):
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        return matrix

    @staticmethod
    def corr_to_cov(corr, std):
        return np.diag(std) @ corr @ np.diag(std)

    @staticmethod
    def cov_to_corr_std(cov):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        return corr, std


class MeanVarianceOptimiser(object):
    """
    Base class for analytical Mean Variance Optimisation

    """

    def __init__(self, mu, cov=None, corr=None, std=None):
        self.target_mu = None
        self.allocation = []
        self.risk = 0

        self.mu = np.array(mu)
        self.covariance_object = Covariance(cov=cov, corr=corr, std=std)
        self.cov = self.covariance_object.cov
        self.cov_inv = np.linalg.inv(self.cov)
        self.ones = np.array(len(mu) * [1])
        self.a, self.b, self.c = self.compute_abc()

    def set_target_mu(self, target_mu):
        self.target_mu = target_mu

    def compute_abc(self):
        a = self.ones.transpose() @ (self.cov_inv @ self.ones)
        b = self.mu.transpose() @ (self.cov_inv @ self.ones)
        c = self.mu.transpose() @ (self.cov_inv @ self.mu)
        return a, b, c

    def compute_optimal_allocation(self):
        raise NotImplementedError

    def compute_portfolio_risk(self):
        raise NotImplementedError


class MeanVarianceOptimiserRiskyOnly(MeanVarianceOptimiser):
    """
    For a given target return m, this class computes the portfolio allocation among risky assets
    that minimises volatility whilst meeting this target and being fully invested

    As an optimisation problem, w solves
    w = argmin 1/2 <w, Omega w>, s.t <mu, w> = m, <1, w> = 1

    Letting lambda be the coefficient of the return constraint and gamma the coefficient of the investment constraint:
    w = Omega^{-1}(lambda x mu + gamma x 1)

    """

    def __init__(self, mu, cov=None, corr=None, std=None):
        super(MeanVarianceOptimiserRiskyOnly, self).__init__(mu, cov=cov, corr=corr, std=std)

        self._lambda_coefficient = None
        self._gamma_coefficient = None

        # the determinant of the matrix used in calculating lambda and gamma
        self.det = self.a * self.c - self.b ** 2

    def compute_lambda_coefficient(self):
        return (self.a * self.target_mu - self.b) / self.det

    def compute_gamma_coefficient(self):
        return (self.c - self.b * self.target_mu) / self.det

    @property
    def lambda_coefficient(self):
        self._lambda_coefficient = self.compute_lambda_coefficient()
        return self._lambda_coefficient

    @property
    def gamma_coefficient(self):
        self._gamma_coefficient = self.compute_gamma_coefficient()
        return self._gamma_coefficient

    def compute_optimal_allocation(self):
        if self.target_mu is None:
            print('Returning Global minimum variance portfolio as no target return level was specified')
            self.set_target_mu(self.b / self.a)
        self.allocation = self.cov_inv @ (self.lambda_coefficient * self.mu + self.gamma_coefficient * self.ones)
        return self.allocation

    def compute_portfolio_risk(self, compute_allocation=True):
        if compute_allocation:
            self.compute_optimal_allocation()
        self.risk = np.sqrt(self.allocation @ (self.cov @ self.allocation))
        return self.risk


class MeanVarianceOptimiserWithRiskFreeAsset(MeanVarianceOptimiser):
    """
    For a given target return m, this class computes the portfolio allocation among risky assets and a risk free asset,
    whose return r is to be specified, with minimal volatility achieving the target return

    As an optimisation problem, w solves
    w = argmin 1/2 <w, Omega w>, s.t <mu - r1, w> = m - r

    The explicit solution is gven by
    w = Omega^{-1}(mu - r x 1) / <mu - r x 1, Omega^{-1}(mu - r x 1)>

    Note that the denominator can be written as C - 2Br + Ar^2

    """

    def __init__(self, mu, cov=None, corr=None, std=None, rfr=0):
        super(MeanVarianceOptimiserWithRiskFreeAsset, self).__init__(mu, cov=cov, corr=corr, std=std)
        # risk-free rate - rfr
        self.rfr = rfr
        self.target_mu = None
        self.denominator = self.c - 2 * self.b * self.rfr + self.a * self.rfr ** 2

    def compute_optimal_allocation(self):
        if self.target_mu is None:
            print('Returning Tangency Portfolio as no target return level was specified')
            self.set_target_mu((self.c - self.rfr * self.b) / (self.b - self.rfr * self.a))
        self.allocation = (self.target_mu - self.rfr) * (
                self.cov_inv @ (self.mu - self.rfr * self.ones)) / self.denominator
        self.rfr_allocation = 1 - self.allocation.sum()

    def compute_portfolio_risk(self):
        self.compute_optimal_allocation()
        self.risk = np.sqrt(self.allocation @ (self.cov @ self.allocation))
        return self.risk


class MeanVarianceUtils(object):
    def __init__(self, mvo, enforce_min_return=False):
        if isinstance(mvo, MeanVarianceOptimiser):
            self.mvo = mvo
            if hasattr(mvo, 'rfr'):
                self.rfr = mvo.rfr
            else:
                self.rfr = 0
        else:
            raise TypeError('mvo must inherit from MeanVarianceOptimiser')
        self.enforce_minimum_return = enforce_min_return

    def compute_risk_return(self, returns):
        results = {}
        if self.enforce_minimum_return:
            if self.rfr:
                minimum_return = self.rfr
            else:
                minimum_return = self.mvo.b / self.mvo.a
        else:
            minimum_return = -np.inf
        for rtn in returns:
            if not self.enforce_minimum_return or rtn > minimum_return:
                self.mvo.set_target_mu(rtn)
                self.mvo.compute_optimal_allocation()
                results[rtn] = self.mvo.compute_portfolio_risk()
        return results

    def plot_efficient_frontier(self, max_return=0.20, min_return=0.00, granularity=0.001, plot_tangency=False):
        effective_min = -(int(-min_return / granularity) + 1)
        effective_max = int(max_return / granularity) + 1
        results = self.compute_risk_return([granularity * x for x in range(effective_min, effective_max)])
        plt.scatter(results.values(), results.keys(), marker='.')

        mvo = self.mvo

        # add global minimum variance portfolio
        plt.scatter(1 / np.sqrt(mvo.a), mvo.b / mvo.a, color='orange')

        # add tangency portfolio
        if self.rfr and plot_tangency:
            tangency_mu = (mvo.c - mvo.rfr * mvo.b) / (mvo.b - mvo.rfr * mvo.a)
            mvo.set_target_mu(tangency_mu)
            plt.scatter(mvo.compute_portfolio_risk(), tangency_mu, c='green')
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.title('Mean-Variance Efficient Frontier')


class InvestmentUniverse(object):
    def __init__(self, mu, cov=None, corr=None, std=None):
        self.size = len(mu)
        self.mu = np.array(mu)
        self.covariance_object = Covariance(cov=cov, corr=corr, std=std)
        self.cov = self.covariance_object.cov
        self.mvo = MeanVarianceOptimiserRiskyOnly(mu=mu, cov=cov, corr=corr, std=std)


class RandomPortfolio(object):
    def __init__(self, weights, universe):
        self.universe = universe
        self.weights = weights

    def compute_portfolio_risk(self):
        return np.sqrt(self.weights @ self.universe.cov @ self.weights)

    def compute_return(self):
        return self.universe.mu @ self.weights


class RandomPortfolioWithRFR(RandomPortfolio):
    def __init__(self, weights, universe, rfr):
        super(RandomPortfolioWithRFR, self).__init__(weights=weights, universe=universe)
        self.rfr = rfr

    def compute_return(self):
        return (1 - self.weights.sum()) * self.rfr + self.universe.mu @ self.weights


class RandomPortfolioGenerator(object):
    def __init__(self, universe, min_return=-0.5, max_return=0.5, max_risk=1):
        self.universe = universe
        self.min_return = min_return
        self.max_return = max_return
        self.max_risk = max_risk

    def generate_random_portfolios(self, number_samples):
        raise NotImplementedError

    def get_sampled_opportunity_set(self, random_portfolios):
        results = {}
        for ptf in random_portfolios:
            results[ptf.compute_return()] = ptf.compute_portfolio_risk()
        return results

    def plot_opportunity_set(self, random_portfolios):
        samples = self.get_sampled_opportunity_set(random_portfolios)
        mvo_utils = MeanVarianceUtils(self.universe.mvo)
        mvo_utils.plot_efficient_frontier(max_return=self.max_return, min_return=self.min_return)
        plt.scatter(samples.values(), samples.keys(), marker='.')


class NormalRandomPortfolioGenerator(RandomPortfolioGenerator):
    def __init__(self, universe, min_return=-0.5, max_return=0.5, max_risk=1, distribution_mean=0, distribution_std=1):
        super(NormalRandomPortfolioGenerator, self).__init__(universe=universe, min_return=min_return,
                                                             max_return=max_return, max_risk=max_risk)
        self.distribution_mean = distribution_mean
        self.distribution_std = distribution_std

    def generate_random_portfolios(self, number_samples):
        samples = []
        counter = 0
        while counter < number_samples:
            counter += 1
            x = np.random.normal(loc=self.distribution_mean, scale=self.distribution_std, size=self.universe.size)
            ptf = RandomPortfolio(weights=x / x.sum(), universe=self.universe)
            ptf_rtn = ptf.compute_return()
            if ptf_rtn <= self.max_return and ptf_rtn >= self.min_return and ptf.compute_portfolio_risk() < self.max_risk:
                samples += [ptf]
        return samples
