from MonteCarlo import Payoff, PathGenerator, MonteCarloPricer
from Helpers import PricingHelpers, AnalyticalHelpers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
import pandas as pd
import numpy as np
import os


# Script will take 5-10 minutes to run first time round as simulated paths are being generated


def plot_random_paths(s, r, d, t, sigma, num_paths, num_time_steps, frequency_name):
    random_draws = np.random.normal(0, 1, (num_time_steps, num_paths))
    path_generator = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma, randomnes=random_draws)

    handles = []

    plt.figure(figsize=(6, 4))
    for path_type in PATH_TYPES:
        colour = PATH_COLOURS.get(path_type)
        save_name = PATH_SAVE_NAMES.get(path_type)
        random_path = path_generator.generate_paths(path_type=path_type,
                                                    num_paths=num_paths, num_time_steps=num_time_steps)[0]
        plt.plot(random_path, color=colour)
        handles.append(mpatches.Patch(color=colour, label=save_name))

    plt.legend(handles=handles, loc=2)
    plt.title(f'{frequency_name} Path Simulations')
    plt.savefig(f'Results/{frequency_name.lower()}_path_simulations.png')
    plt.close()


def plot_mean_relative_errors_from_repeats(mean_data, ts_list, path_type, analytical_value):
    idx = mean_data.index
    path_cols = sorted(list(set([x.split('_')[0] for x in idx])))
    mean_data['paths'] = [x.split('_')[0] for x in idx]
    mean_data['time'] = [x.split('_')[-1] for x in idx]
    data = mean_data.pivot(index='time', columns='paths')
    data.to_csv(f'Results/{path_type}_mean_error.csv')
    plt.figure(figsize=(6, 4))
    for time_steps in ts_list:
        plt.scatter(path_cols, 100 * data.loc[str(time_steps)] / analytical_value)
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Error [%]')
    plt.title(f'{" ".join(path_type.split("_")).title()} MC Error')
    plt.legend([f'{str(x)} Time Steps' for x in ts_list], loc=1)
    plt.savefig(f'Results/{path_type}_path_number_impact.png')
    plt.close()


def plot_std_error_from_repeats(std_error, ts_list, path_type, analytical_value):
    idx = std_error.index
    path_cols = sorted(list(set([x.split('_')[0] for x in idx])))
    std_error['paths'] = [x.split('_')[0] for x in idx]
    std_error['time'] = [x.split('_')[-1] for x in idx]
    data = std_error.pivot(index='time', columns='paths')

    log_data = np.log10(data).transpose()
    plt.figure(figsize=(6, 4))
    for ts in ts_list:
        plt.scatter(path_cols, log_data[str(ts)])
    plt.xlabel('Number of Paths')
    plt.ylabel('Log Standard Deviation')
    plt.title(f'{" ".join(path_type.split("_")).title()} MC Error Variation')
    plt.legend([f'{str(x)} Time Steps' for x in ts_list], loc=1)
    plt.savefig(f'Results/{path_type}_std_log_path_number_impact.png')
    plt.close()


def repeats_plots(repeats, analytical_value, ts_list):
    repeats_mc = repeats[[c for c in repeats.columns if 'time' not in c]]
    repeats_mc.columns = [f'10^{int(np.log10(int(c.split("_")[0])))}_{c.split("_")[1]}' for c in repeats_mc.columns]

    error = repeats_mc - analytical_value
    absolute_error = abs(error)
    mean_data = pd.DataFrame(absolute_error.mean())
    plot_mean_relative_errors_from_repeats(mean_data=mean_data, ts_list=ts_list, path_type=path_type,
                                           analytical_value=analytical_value)

    std_error = pd.DataFrame(repeats_mc.std())
    plot_std_error_from_repeats(std_error=std_error, ts_list=ts_list, path_type=path_type,
                                analytical_value=analytical_value)


def plot_running_mc_convergence(repeats, running_values, std,
                                num_paths, num_time_steps, num_repeats=10, skip_paths=500, title=''):
    plt.figure(figsize=(6, 4))
    new_idx = running.index[skip_paths:]
    for idx in range(1, num_repeats + 1):
        plt.plot(running_values[idx].iloc[skip_paths:], color='indianred')
    plt.plot((num_paths - skip_paths) * [c_cn_fair], color='orange')

    plt.plot([c_cn_fair + std / np.sqrt(k) for k in new_idx], '--', color='steelblue')
    plt.plot([c_cn_fair - std / np.sqrt(k) for k in new_idx], '--', color='steelblue')
    plt.plot([c_cn_fair + 2 * std / np.sqrt(k) for k in new_idx], '--', color='teal')
    plt.plot([c_cn_fair - 2 * std / np.sqrt(k) for k in new_idx], '--', color='teal')

    two_std_up_patch = mpatches.Patch(color='teal', label='$\pm2\sigma/\sqrt{N}$')
    std_up_patch = mpatches.Patch(color='steelblue', label='$\pm\sigma/\sqrt{N}$')
    value_patch = mpatches.Patch(color='orange', label='Theoretical Value')
    handle = [two_std_up_patch, std_up_patch, value_patch]
    plt.legend(handles=handle, loc=1)

    plt.title(f'{title}')
    plt.savefig(f'Results/10^{int(np.log10(num_paths))}_{num_time_steps}_{title}_mc_convergence.png')
    plt.close()


def get_payoff_standard_deviation(strike, r, d, t, sigma, payoff, num_paths, num_time_steps, path_type):
    pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
    sample_paths = pg.generate_paths(path_type=path_type, num_paths=num_paths, num_time_steps=num_time_steps)[0]
    pricer = MonteCarloPricer(strike=strike, r=r, d=d, t=t, sample_paths=sample_paths, payoff=payoff,
                              return_std_error=False)
    return pricer.compute_payoff_standard_deviation()


def compute_required_num_paths(std, accuracy, probability):
    return norm.ppf((1 - probability) / 2) ** 2 * std ** 2 / accuracy ** 2


def plot_payoff_standard_deviation(stds, parameter):
    standard_deviations = pd.DataFrame(list(stds.values()), index=[str(x) for x in stds.keys()], columns=['std'])
    standard_deviations.index = [str(round(float(x), 2)) for x in standard_deviations.index]
    plt.figure(figsize=(6, 4))
    plt.bar(standard_deviations.index, standard_deviations['std'])
    plt.xlabel('Price')
    plt.savefig(f'Results/std_by_{parameter}.png')
    plt.close()


def risk_neutral_itm_probability(s, k, r, d, t, sigma):
    return norm.cdf((np.log(s / k) + (r - d - sigma ** 2 / 2) * t) / (sigma * np.sqrt(t)))


def risk_neutral_itm_variance(s, k, r, d, t, sigma):
    p = risk_neutral_itm_probability(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
    return p * (1 - p)


plt.style.use('ggplot')
PATH_TYPES = ['euler_maruyama', 'milstein', 'log_normal']
PATH_SAVE_NAMES = {'euler_maruyama': 'Euler-Maruyama', 'milstein': 'Milstein', 'log_normal': 'Lognormal'}
PATH_COLOURS = {'euler_maruyama': 'indianred', 'milstein': 'teal', 'log_normal': 'steelblue'}

if not os.path.exists('Results/'):
    os.mkdir('Results/')

# Exam Example
s = 100
k = 100
r = 0.05
d = 0.00
t = 1
sigma = 0.20

c_cn_payoff = Payoff.european_cash_or_nothing_binary_call
p_cn_payoff = Payoff.european_cash_or_nothing_binary_put

p_cn_fair = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_put(s, k, r, d, t, sigma)
c_cn_fair = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_call(s, k, r, d, t, sigma)

# sense check
run_check = False
if run_check:
    c_parity = AnalyticalHelpers.binary_cash_or_nothing_put_to_call(p_cn_fair, r, t)
    p_parity = AnalyticalHelpers.binary_cash_or_nothing_call_to_put(c_cn_fair, r, t)
    assert (abs(p_cn_fair - p_parity) < 1e-5)
    assert (abs(c_cn_fair - c_parity) < 1e-5)
    pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
    gbm_paths, runtime = pg.log_normal(num_time_steps=50, num_paths=100000)
    c_mc, c_running_values, c_abs_e, c_rel_e, c_se = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t,
                                                                                        paths=gbm_paths,
                                                                                        payoff=c_cn_payoff,
                                                                                        analytical_value=c_cn_fair)
    p_mc, p_running_values, p_abs_e, p_rel_e, p_se = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t,
                                                                                        paths=gbm_paths,
                                                                                        payoff=p_cn_payoff,
                                                                                        analytical_value=p_cn_fair)

# generate illustrative paths
num_paths = 100
frequencies = {'Monthly': 12, 'Daily': 250}
for frequency in frequencies:
    if not os.path.exists(f'Results/{frequency.lower()}_path_simulations.png'):
        num_time_steps = frequencies.get(frequency)
        plot_random_paths(s=s, r=r, d=d, t=t, sigma=sigma, num_paths=num_paths, num_time_steps=num_time_steps,
                          frequency_name=frequency)

# compute theoretical option values
file_name = 'Results/analytical_option_values.csv'
if not os.path.exists(file_name):
    analytical_helper = AnalyticalHelpers()
    values = {}
    payoff_methods = [f for f in analytical_helper.__dir__() if f.startswith('analytical')]
    for payoff_method in payoff_methods:
        values[payoff_method] = analytical_helper.__getattribute__(payoff_method)(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
    pd.DataFrame(list(values.values()), index=list(values.keys()), columns=['Value']).to_csv(file_name)

# repeat monte carlo pricing for mean error analysis
timesteps = {'milstein': [1, 5, 10, 20], 'log_normal': [1, 5, 10], 'euler_maruyama': [1, 3, 5, 10, 20, 50]}

num_repeats = 50
payoff = c_cn_payoff

for path_type in PATH_TYPES:
    ts_list = timesteps.get(path_type)
    save_name = f'Results/{num_repeats}_repeats_{path_type}_timesteps_{"_".join([str(ts) for ts in ts_list])}.csv'

    if not os.path.exists(save_name):
        print(f'Creating {path_type} price paths')
        repeats = pd.DataFrame()
        for num_paths in [10 ** k for k in [2, 3, 4, 5, 6]]:
            for time_steps in ts_list:
                print(num_paths, time_steps)
                column_name = f'{num_paths}_{time_steps}_{path_type}'
                repeats = repeats.transpose().append(
                    PricingHelpers.repeated_pricing(s=s, k=k, r=r, d=d, t=t, sigma=sigma, payoff=payoff,
                                                    path_type=path_type, num_paths=num_paths, num_timesteps=time_steps,
                                                    repeats=num_repeats,
                                                    column_name=column_name).transpose()).transpose()
        repeats.to_csv(f'{save_name}')

for path_type in PATH_TYPES:
    ts_list = timesteps.get(path_type)
    save_name = f'Results/{num_repeats}_repeats_{path_type}_timesteps_{"_".join([str(ts) for ts in ts_list])}.csv'
    repeats = pd.read_csv(save_name, index_col=0)

    if not os.path.exists(f'Results/{path_type}_path_number_impact.png'):
        repeats_plots(repeats=repeats, analytical_value=c_cn_fair, ts_list=ts_list)

# compute standard deviation of payoff
path_type = 'euler_maruyama'
std = {}
file_name = f'Results/{path_type}_std_cn_call.csv'
if not os.path.exists(file_name):
    for ts in timesteps.get(path_type):
        if ts not in std:
            std[ts] = {}
        for n_pths in [10 ** k for k in range(2, 7)]:
            std[ts][n_pths] = get_payoff_standard_deviation(strike=k, r=r, d=d, t=t, sigma=sigma, payoff=c_cn_payoff,
                                                            num_paths=n_pths, num_time_steps=ts, path_type=path_type)
    pd.DataFrame(std).to_csv(f'Results/{path_type}_std_cn_call.csv')

# have payoff standard deviation available for below methods
std_data = pd.read_csv(file_name, index_col=0)
std = std_data.iloc[-1, -1]

new_file_name = 'Results/required_paths_for_accuracy.csv'
if not os.path.exists(new_file_name):
    required_paths = {}
    accuracy_levels = [0.02, 0.01, 0.005, 0.0025, 0.001, 0.0001]
    cash_accuracy = [c_cn_fair * accuracy for accuracy in accuracy_levels]
    for accuracy in accuracy_levels:
        required_paths[accuracy] = int(compute_required_num_paths(std, accuracy=accuracy, probability=0.99)) + 1
    accuracy_data = pd.DataFrame(list(required_paths.values()), index=list(required_paths.keys()),
                                 columns=['Required Paths'])
    accuracy_data['Cash Accuracy'] = cash_accuracy
    accuracy_data.to_csv(new_file_name)

# running value plots
path_type = 'euler_maruyama'
num_paths = 100000
num_repeats = 5
skip_paths = 500
for num_time_steps in [5, 50]:
    title = f"{path_type.title().replace('_', '-')} Convergence - {num_time_steps} Time Steps"

    pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
    em_paths, runtime = pg.euler_maruyama(num_time_steps=num_time_steps, num_paths=num_paths)

    repeats, running = PricingHelpers.repeated_pricing(s=s, k=k, r=r, d=d, t=t, sigma=sigma, payoff=c_cn_payoff,
                                                       path_type=path_type, num_paths=num_paths,
                                                       num_timesteps=num_time_steps, repeats=num_repeats,
                                                       return_running_values=True)

    plot_running_mc_convergence(repeats=repeats, running_values=running, std=std, num_repeats=num_repeats,
                                num_paths=num_paths, num_time_steps=num_time_steps, skip_paths=skip_paths, title=title)

path_type = 'euler_maruyama'
num_paths = 100000
num_repeats = 50
time_steps = 50

# varying moneyness
varying_param = 'moneyness'
save_name = f'Results/{varying_param}_repeats_{path_type}.csv'
if not os.path.exists(save_name):
    repeats = pd.DataFrame()
    for s in [100 + 10 * x for x in range(-5, 6)]:
        column_name = f'{path_type}_stock_{s}_strike_{k}'
        repeats = repeats.transpose().append(
            PricingHelpers.repeated_pricing(s=s, k=k, r=r, d=d, t=t, sigma=sigma, payoff=c_cn_payoff,
                                            path_type=path_type, num_paths=num_paths, num_timesteps=time_steps,
                                            repeats=num_repeats,
                                            column_name=column_name).transpose()).transpose()
    repeats.to_csv(f'{save_name}')

repeats = pd.read_csv(save_name, index_col=0)
values = {}
stds = {}
for s in [100 + 10 * x for x in range(-5, 6)]:
    values[s] = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_call(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
    pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
    paths = pg.generate_paths(path_type=path_type, num_paths=num_paths, num_time_steps=num_time_steps)[0]
    mc, running, abs_e, rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=paths,
                                                                      payoff=c_cn_payoff, analytical_value=values[s])
    stds[s] = running.std()[0]

plot_payoff_standard_deviation(stds, parameter='s')

repeats_mc = repeats[[c for c in repeats.columns if 'time' not in c]]
repeats_mc.columns = [int(c.split('_')[3]) for c in repeats_mc.columns]
analytical_values = pd.DataFrame(np.array([list(values.values()) for _ in range(len(repeats_mc))]),
                                 columns=values.keys())
error = repeats_mc - analytical_values
absolute_error = abs(error).mean()
plt.figure(figsize=(6, 4))
plt.bar([str(x) for x in absolute_error.index], absolute_error)
plt.title('Mean Absolute Error by Price')
plt.xlabel('Price')
plt.yticks([])
plt.savefig(f'Results/mean_error_by_{varying_param}.png')
plt.close()

if not os.path.exists(f'Results/cn_call_payoff_properties.png'):
    rnp = {}
    rnv = {}
    for s in [50 + 0.1 * x for x in range(1001)]:
        rnv[s] = risk_neutral_itm_variance(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
    rnv = pd.DataFrame(list(rnv.values()), index=list(rnv.keys()), columns=['Variance'])
    for s in [50 + 0.1 * x for x in range(1001)]:
        rnp[s] = risk_neutral_itm_probability(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
    rnp = pd.DataFrame(list(rnp.values()), index=list(rnp.keys()), columns=['ITM Probability'])
    plt.plot(rnp)
    plt.plot(rnv)
    plt.xlabel('Price')
    plt.title('Cash or Nothing Call Payoff Properties')
    plt.legend(['ITM Probability', 'Payoff Variance'], loc=2)
    plt.savefig(f'Results/cn_call_payoff_properties.png')
    plt.close()

# varying volatility
# out/at/in the money
for stock in [60, 100, 140]:
    print(stock)
    varying_param = f'volatility_stock_{stock}'
    save_name = f'Results/{varying_param}_repeats_{path_type}.csv'
    if not os.path.exists(save_name):
        repeats = pd.DataFrame()
        for sigma in [0.05 + 0.05 * x for x in range(10)]:
            column_name = f'{path_type}_{varying_param}_{sigma}'
            repeats = repeats.transpose().append(
                PricingHelpers.repeated_pricing(s=s, k=k, r=r, d=d, t=t, sigma=sigma, payoff=c_cn_payoff,
                                                path_type=path_type, num_paths=num_paths, num_timesteps=time_steps,
                                                repeats=num_repeats,
                                                column_name=column_name).transpose()).transpose()
        repeats.to_csv(f'{save_name}')

if not os.path.exists('Results/std_by_volatility_stock.csv'):
    all_stds = {}
    for stock in [60, 100, 140]:
        varying_param = f'volatility_stock_{stock}'
        save_name = f'Results/{varying_param}_repeats_{path_type}.csv'
        repeats = pd.read_csv(save_name, index_col=0)
        values = {}
        stds = {}
        for sigma in [0.05 + 0.05 * x for x in range(10)]:
            values[sigma] = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_call(s=s, k=k, r=r, d=d, t=t,
                                                                                              sigma=sigma)
            pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
            paths = pg.generate_paths(path_type=path_type, num_paths=num_paths, num_time_steps=num_time_steps)[0]
            mc, running, abs_e, rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=paths,
                                                                              payoff=c_cn_payoff,
                                                                              analytical_value=values[sigma])
            stds[sigma] = running.std()[0]
        all_stds[stock] = stds
    result = pd.DataFrame(all_stds)
    result.to_csv('Results/std_by_volatility_stock.csv')
result = pd.read_csv('Results/std_by_volatility_stock.csv', index_col=0)
result.index = [str(round(float(x), 2)) for x in result.index]
idx = result.index

for c in result.columns:
    plt.bar(idx, result[c])
plt.legend(result.columns)
plt.xlabel('Volatility')
plt.yticks([])
plt.title('Impact of Volatility on Payoff Variation')
plt.savefig('Results/payoff_std_by_volatility.png')
plt.close()

# PricingHelpers.plot_by_spot_and_time(s=s, k=k, r=r, d=d, sigma=sigma,
#                                      payoff=AnalyticalHelpers.cash_or_nothing_european_call_delta)
# PricingHelpers.plot_by_spot_and_time(s=s, k=k, r=r, d=d, sigma=sigma,
#                                      payoff=AnalyticalHelpers.cash_or_nothing_european_call_gamma)

# Haug example 1 - vanillas

# s = 60
# k = 65
# r = 0.08
# d = 0
# t = 0.25
# sigma = 0.3
#
# c = AnalyticalHelpers.analytical_european_call(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
# p = AnalyticalHelpers.analytical_european_put(s=s, k=k, r=r, d=d, t=t, sigma=sigma)
# assert (abs(c - 2.1334) < 1e-4)
# assert abs(c - AnalyticalHelpers.put_to_call(p, s, k, r, d, t)) < 1e-8
# assert abs(p - AnalyticalHelpers.call_to_put(c, s, k, r, d, t)) < 1e-8
#
# p_c = Payoff.european_call
# p_p = Payoff.european_put
# pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
# gbm_paths, runtime = pg.log_normal(num_time_steps=1, num_paths=1000000)
# c_mc, c_running, c_abs_e, c_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                           payoff=p_c, analytical_value=c)
# p_mc, p_running, p_abs_e, p_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                           payoff=p_p, analytical_value=p)

# Haug example 2 - binary cash or nothing

# s = 100
# k = 80
# r = 0.06
# d = 0.06
# t = 0.75
# sigma = 0.35
# cash_payoff = 1

# analytical
# p = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_put(s, k, r, d, t, sigma, payoff=cash_payoff)
# c = AnalyticalHelpers.analytical_european_cash_or_nothing_binary_call(s, k, r, d, t, sigma, payoff=cash_payoff)
# c_parity = AnalyticalHelpers.binary_cash_or_nothing_put_to_call(p, r, t, payoff=cash_payoff)
# p_parity = AnalyticalHelpers.binary_cash_or_nothing_call_to_put(c, r, t, payoff=cash_payoff)
# assert (abs(p - 0.26710) < 1e-4)
# assert (abs(p - p_parity) < 1e-8)
# assert (abs(c - c_parity) < 1e-8)
#
# # Monte Carlo
# c_p = Payoff.european_cash_or_nothing_binary_call
# p_p = Payoff.european_cash_or_nothing_binary_put
# pg = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
# gbm_paths, runtime = pg.log_normal(num_time_steps=1, num_paths=1000000)
# c_mc, c_running, c_abs_e, c_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                                  payoff=c_p, analytical_value=c)
# p_mc, p_running, p_abs_e, p_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                                  payoff=p_p, analytical_value=p)

# Haug example 3 - binary asset or nothing

# s = 70
# k = 65
# r = 0.07
# d = 0.05
# t = 0.5
# sigma = 0.27
#
# # analytical
# p = AnalyticalHelpers.analytical_european_asset_or_nothing_binary_put(s, k, r, d, t, sigma)
# c = AnalyticalHelpers.analytical_european_asset_or_nothing_binary_call(s, k, r, d, t, sigma)
# c_parity = AnalyticalHelpers.binary_asset_or_nothing_put_to_call(p, s, r, d, t)
# p_parity = AnalyticalHelpers.binary_asset_or_nothing_call_to_put(c, s, r, d, t)
# assert (abs(p - 20.2069) < 1e-4)
# assert (abs(p - p_parity) < 1e-8)
# assert (abs(c - c_parity) < 1e-8)
#
# # Monte Carlo
# c_p = Payoff.european_asset_or_nothing_binary_call
# p_p = Payoff.european_asset_or_nothing_binary_put
# pths = PathGenerator(s=s, r=r, d=d, t=t, sigma=sigma)
# gbm_paths, runtime = pths.log_normal(num_time_steps=1, num_paths=100000)
# c_mc, c_running, c_abs_e, c_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                           payoff=c_p, analytical_value=c)
# p_mc, p_running, p_abs_e, p_rel_e, _ = PricingHelpers.compare_european_mc(k=k, r=r, d=d, t=t, paths=gbm_paths,
#                                                                           payoff=p_p, analytical_value=p)
#
