import Portfolio, RiskMeasures
from helpers import *
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

plt.style.use('ggplot')

os.makedirs('Results/', exist_ok=True)

# Q1/Q2 Data

mu = np.array([0.02, 0.07, 0.15, 0.20])
sigma = np.array([0.05, 0.12, 0.17, 0.25])

corr = np.array([[1, 0.3, 0.3, 0.3],
                 [0.3, 1, 0.6, 0.6],
                 [0.3, 0.6, 1, 0.6],
                 [0.3, 0.6, 0.6, 1]])

# Q1

# calculate weights and allocations
weights = {}
risk = {}

target_mu = 4.5 / 100

for stress_factor in [1, 1.25, 1.5]:
    stressed_corr = stressed_correlation_matrix(corr, stress_factor)
    w, r = compute_risky_only_allocation_and_risk(mu=mu, sigma=sigma, corr=stressed_corr, target_mu=target_mu)
    weights[stress_factor] = w
    risk[stress_factor] = r

# form weights dataframe indexed by stress factor
weights = pd.DataFrame(weights).transpose()
weights.columns = list('ABCD')
risk = pd.Series(risk)
weights['Risk'] = risk

# save
weights.to_csv('Results/q1_allocation_and_risk.csv')

# create and plot random portfolios
universe = Portfolio.InvestmentUniverse(mu=mu, corr=corr, std=sigma)
normal_ptf_generator = Portfolio.NormalRandomPortfolioGenerator(universe=universe)

random_portfolios = normal_ptf_generator.generate_random_portfolios(number_samples=7000)
normal_ptf_generator.plot_opportunity_set(random_portfolios)
plt.savefig('Results/q1_random_portfolio_plot.png')

# Q2

rfrs = [x / 10000 for x in [50, 100, 150, 175]]

weights = {}
risk = {}

for rfr in rfrs:
    w, r = compute_tangency_portfolio_weights_and_risk(mu=mu, corr=corr, sigma=sigma, rfr=rfr)
    weights[rfr] = w
    risk[rfr] = r

weights = pd.DataFrame(weights).transpose()
weights.columns = list('ABCD')
risk = pd.Series(risk)
weights['Risk'] = risk
weights.to_csv('Results/q2_allocation_and_risk.csv')

for rfr in [100 / 10000, 175 / 10000]:
    mvo = Portfolio.MeanVarianceOptimiserRiskyOnly(mu=mu, std=sigma, corr=corr)
    mvo_rf = Portfolio.MeanVarianceOptimiserWithRiskFreeAsset(mu=mu, std=sigma, corr=corr, rfr=rfr)
    util = Portfolio.MeanVarianceUtils(mvo=mvo, enforce_min_return=True)
    util_rf = Portfolio.MeanVarianceUtils(mvo=mvo_rf, enforce_min_return=True)
    tangency_return = (mvo.c - rfr * mvo.b) / (mvo.b - rfr * mvo.a)

    plot_tangency = False

    # determine borders of figure
    max_return = 0.2
    min_return = 0
    if tangency_return > 0:
        max_return = tangency_return
        plot_tangency = True
    else:
        min_return = tangency_return

    # plot frontier and CML
    plt.figure()
    util.plot_efficient_frontier(max_return=max_return, min_return=min_return)
    util_rf.plot_efficient_frontier(max_return=max_return, min_return=min_return, plot_tangency=plot_tangency)
    plt.scatter(0, rfr, marker='x', color='black')

    # hack to handle legend
    legend = ['Risky Only Efficient Frontier', '_', 'True Efficient Frontier - CML', 'GMV Portfolio']
    if rfr < 160 / 10000:
        legend += ['Tangency Portfolio']
    legend += [f'Risk Free Asset - Rate {round(100 * rfr, 2)}%']
    plt.legend(legend)

    plt.title(f'Efficient Frontier - Risk Free Rate {round(100 * rfr, 2)}%')
    plt.savefig(f'Results/q2_efficient_frontier_{round(100 * rfr, 2)}%.png')

# Q3

data_path = 'SP500.csv'
data = pd.read_csv(data_path, index_col=0)
data.index = [dt.datetime.strptime(d, '%d/%m/%Y') for d in data.index]
idx = ['Breaches', '% Breaches', 'Consec. Breaches', 'Cond. Prob. Consec. Breach']
results = pd.DataFrame()

for method in ['Rolling', 'EWMA']:
    if method == 'Rolling':
        var_bt = RiskMeasures.VarBacktestRollingVolatility(price_data=data)
    elif method == 'EWMA':
        var_bt = RiskMeasures.VarBacktestEWMAVolatility(price_data=data)
    else:
        raise BaseException('Not supposed to be here')

    plt.figure()
    var_bt.plot_var_versus_realised_return()
    plt.savefig(f'Results/q3_{method.lower()}_volatility.png')

    # get breaches, pct_breaches, consecutive breaches, conditional probability of consecutive breach
    b, pct_b, c, cp_cb = var_bt.get_headline_figures()
    df = pd.DataFrame([b, 100 * pct_b, c, cp_cb], index=idx).transpose()
    df.index = [method]
    results[method] = df.T[method]
results.to_csv('Results/q3_headline_figures.csv')

# Q4
results = pd.DataFrame()
idx = ['Total', 'Total Rel.', 'Asset', 'Asset Rel', 'Liq.', 'Liq Rel.']

params = [['Technology', 16000000, 0.01, 0.03, -0.0035 / 2, 0.015 / 2],
          ['Gilt 15bp', 40000000, 0, 0.03, -15 / 10000 / 2, 0],
          ['Gilt 125 bp', 40000000, 0, 0.03, -125 / 10000 / 2, 0]]
for param_set in params:
    scenario, notional, mu, sigma, mu_spread, sigma_spread = param_set
    total, total_rel, asset, asset_rel, liq, liq_rel = compute_liquidity_adjusted_var(mu=mu, sigma=sigma,
                                                                                      mu_spread=mu_spread,
                                                                                      sigma_spread=sigma_spread,
                                                                                      notional=notional)
    df = pd.DataFrame([total, total_rel, asset, asset_rel, liq, liq_rel], index=idx).transpose()
    df.index = [scenario]
    results[scenario] = df.T[scenario]
results.to_csv('Results/q4_aggregated_figures.csv')
