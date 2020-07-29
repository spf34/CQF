import Portfolio, RiskMeasures
import numpy as np


def compute_risky_only_allocation_and_risk(mu, sigma, corr, target_mu):
    """

    Args:
        mu (numpy.ndarray): vector of returns
        sigma (numpy.ndarray): vector of volatilities
        corr (numpy.ndarray): matrix of correlations
        target_mu (float): target return

    Returns:
        weights (numpy.ndarray): allocation weights
        risk (float): portfolio volatility
    """

    # create mean-variance optimiser object for risky-only universe
    mvc = Portfolio.MeanVarianceOptimiserRiskyOnly(mu=mu, std=sigma, corr=corr)

    # set target return
    mvc.set_target_mu(target_mu=target_mu)

    # compute optimal allocation
    mvc.compute_optimal_allocation()
    weights = mvc.allocation

    # check solution is feasible
    assert weights.sum() == 1
    assert abs(weights @ np.array(mu) - 0.045) < 1e-10

    # compute portfolio risk
    risk = mvc.compute_portfolio_risk()

    # # sense-check average volatility is higher than portfolio volatility
    # average_vol = weights @ np.array(sigma)
    # print(f'Risk reduction benefit: {risk/average_vol}')

    return weights, risk


def stressed_correlation_matrix(correl, stress_factor=1.25):
    """
    Method to stress NxN correlation matrix.

    Denoting by C, C' the original and stressed correlation matrices:
    C' = stress_factor x C - (stress_factor - 1) x identity, where identity is the NxN identity matrix
    """
    correl = np.array(correl)
    number_assets = len(correl)

    # sense check to ensure we don't end up with correlations outside of [-1, 1]
    pure_correl = correl - np.eye(number_assets)
    if pure_correl.max().max() > 1 / stress_factor:
        raise BaseException('Stressing will result in correlations greater than 1')
    if pure_correl.min().min() < -1 / stress_factor:
        raise BaseException('Stressing will result in correlations less than -1')

    return stress_factor * correl - (stress_factor - 1) * np.eye(number_assets)


def compute_tangency_portfolio_weights_and_risk(mu, sigma, corr, rfr):
    # create mean-variance optimiser object for universe with risk-free asset
    mvo = Portfolio.MeanVarianceOptimiserWithRiskFreeAsset(mu=mu, std=sigma, corr=corr, rfr=rfr)

    # compute tangency portfolio and risk
    mvo.compute_optimal_allocation()
    weights = mvo.allocation
    risk = mvo.compute_portfolio_risk()

    # check fully invested
    assert abs(weights.sum() - 1) < 1e-10

    # # sense-check average volatility is higher than portfolio volatility
    # average_vol = weights @ np.array(sigma)
    # print(f'Risk reduction benefit: {risk/average_vol}')

    return weights, risk


def compute_liquidity_adjusted_var(mu, sigma, mu_spread, sigma_spread, notional, days=1, percentile=99):
    var = RiskMeasures.ValueAtRisk(volatility=sigma, expected_return=mu, notional=notional)
    var_spread = RiskMeasures.ValueAtRisk(volatility=sigma_spread, expected_return=mu_spread, notional=notional)
    ptf_var = var.get_notional_simple_percentile_VaR(days=days, percentile=percentile)
    spread_var = var_spread.get_notional_simple_percentile_VaR(days=days, percentile=percentile)
    lvar = ptf_var + spread_var
    lvar_fraction = lvar / notional
    ptf_fraction = ptf_var / lvar
    spread_fraction = 1 - ptf_fraction
    return lvar, lvar_fraction, ptf_var, ptf_fraction, spread_var, spread_fraction
