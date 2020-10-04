import numpy as np
from scipy.stats import norm, skew, kurtosis, pearsonr
from statistics import variance
import pandas as pd
import math
import matplotlib.pyplot as plt

def rolling_sharpe(returns_trial, rf=0, days=252, window=126):
    if type(returns_trial) == dict:
        returns_trial = returns_trial[0]
    volatility = returns_trial.rolling(window).std()
    mean = returns_trial.rolling(window).mean()
    sharpe_ratio = (mean - rf) / volatility
    return (sharpe_ratio * np.sqrt(days)).dropna()

def sharpe(returns_trial, rf=0, days=252):
    if type(returns_trial) == dict:
        returns_trial = returns_trial[0]
    volatility = returns_trial.std()
    sharpe_ratio = (returns_trial.mean() - rf) / volatility
    return sharpe_ratio * np.sqrt(days)

def information_ratio(returns_trial, benchmark_returns=0, days=252):
    if type(returns_trial) == dict:
        returns_trial = returns_trial[0]
    return_difference = returns_trial - benchmark_returns
    volatility = return_difference.std() * np.sqrt(days)
    information_ratio = return_difference.mean() / volatility
    return information_ratio

def PSR(returns_trial, sharpe_ratio, sharpe_ratio_benchmark=0):
    if type(returns_trial) == dict:
        returns_trial = returns_trial[0]
    third_moment = skew(returns_trial)
    forth_moment = kurtosis(returns_trial)
    sample_length = len(returns_trial)
    psr = norm.cdf((sharpe_ratio - sharpe_ratio_benchmark) * (sample_length - 1) ** 0.5 / (1 - sharpe_ratio * third_moment + sharpe_ratio ** 2 * (forth_moment - 1) / 4) ** 0.5)
    return psr, third_moment, forth_moment, sample_length

def calendar_resample(returns, chunk_size, days):
    df_dict = {}
    sr_dict = {}
    nSubset = len(returns) // chunk_size
    for n in range(chunk_size):
        returns_temp = returns.iloc[n * nSubset:(n + 1) * nSubset]
        returns_temp = returns_temp.reset_index(drop=True)
        df_dict[n] = returns_temp
        sr_dict[n] = sharpe(returns_temp, days=days)
    return df_dict, sr_dict

def expected_max_sharpe_ratio(n_independent_trials, sr_dict):
    trials_standard_dev = np.sqrt(variance(sr_dict.values()))
    estimated_sharpe_ratio_benchmark = trials_standard_dev * (1 - np.euler_gamma) * norm.ppf(1 - 1. / n_independent_trials) + np.euler_gamma * norm.ppf(1 - 1. / (n_independent_trials * np.e))
    return estimated_sharpe_ratio_benchmark, trials_standard_dev, n_independent_trials

def implied_num_independent_trials(return_trials):
    rho_dict = {}
    M = len(return_trials)
    for i in range(M):
        for j in range(1, M):
            if j <= i:
                continue
            ret_loop = pd.concat([return_trials[i], return_trials[j]], axis=1)
            ret_loop = ret_loop.dropna()
            ret_loop.columns = ['ret1', 'ret2']
            rho_dict[str(i) + '_' + str(j)] = pearsonr(ret_loop['ret1'], ret_loop['ret2'])[0]
    rho_estimate = (2 * np.sum(list(rho_dict.values()))) / (M * (M - 1))

    n_independent_trials = math.floor(rho_estimate + (1 - rho_estimate) * M)
    return n_independent_trials

def DSR(return_trials, sharpe_ratio, days=1, independent_trials=False, chunk_size=None):
    if chunk_size != None:
        return_trials = return_trials[0]
        return_trials, sr_dict = calendar_resample(return_trials, chunk_size, days)
    else:
        sr_dict = {}
        for n in range(len(return_trials)):
            sr_dict[n] = sharpe(return_trials[n], days=days)
    if independent_trials == False:
        n_independent_trials = implied_num_independent_trials(return_trials)
    estimated_sharpe_ratio_benchmark, trials_standard_dev, n_independent_trials = expected_max_sharpe_ratio(n_independent_trials, sr_dict)
    return PSR(return_trials, sharpe_ratio, estimated_sharpe_ratio_benchmark), estimated_sharpe_ratio_benchmark, return_trials, sr_dict,  trials_standard_dev, n_independent_trials

def minTRL(return_trials, sharpe_ratio, sharpe_ratio_benchmark=0, prob=0.95):
    if type(return_trials) == dict:
        return_trials = return_trials[0]
    return (1 + ( 1 - skew(return_trials) * sharpe_ratio + (kurtosis(return_trials) - 1) / 4. * sharpe_ratio ** 2) * (norm.ppf(prob) / (sharpe_ratio - sharpe_ratio_benchmark)) ** 2)
