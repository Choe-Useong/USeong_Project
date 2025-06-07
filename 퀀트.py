import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.optimize import minimize

def download_prices(tickers, start, end):
    price = yf.download(tickers, start=start, end=end)['Close']
    return price.dropna()

def compute_log_returns(price_df, scale_factor=1):
    return np.log(price_df).diff().dropna() * scale_factor

def estimate_dynamic_mu(returns_df, level='local level', ar=1):
    filtered_mu = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    filtered_var = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    results = {}
    for ticker in returns_df.columns:
        model = UnobservedComponents(returns_df[ticker], level=level, autoregressive=ar)
        res = model.fit(disp=False, maxiter=100)
        filtered_mu[ticker] = res.filtered_state[0]
        filtered_var[ticker] = res.filtered_state_cov[0, 0, :]
        results[ticker] = res
    return filtered_mu, filtered_var, results

def shrink_mean(mu_df, var_df=None, lam=0.0, method='manual'):
    if method == 'manual':
        mean_series = mu_df.mean(axis=1)
        return mu_df.mul(1 - lam, axis=0).add(mean_series.mul(lam), axis=0)

    elif method == 'auto':
        if var_df is None:
            raise ValueError("자동 쉬링크에는 var_df가 필요합니다.")

        T, N = mu_df.shape
        mu_shrunk_df = pd.DataFrame(index=mu_df.index, columns=mu_df.columns)

        for t in range(T):
            mu_t = mu_df.iloc[t]
            mu_bar = mu_t.mean()
            var_t = var_df.iloc[t].mean()
            diff2 = np.sum((mu_t - mu_bar) ** 2)
            diff2 = max(diff2, 1e-6)
            lam_opt = max(0, min(1.0, ((N - 3) * var_t) / diff2))
            mu_shrunk_df.iloc[t] = (1 - lam_opt) * mu_t + lam_opt * mu_bar

        return mu_shrunk_df

    else:
        raise ValueError("method는 'manual' 또는 'auto'만 가능합니다.")

def constant_correlation_target(S):
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    N = S.shape[0]
    mean_corr = (corr.sum() - N) / (N * (N - 1))
    T = mean_corr * np.outer(std, std)
    np.fill_diagonal(T, std ** 2)
    return T

def ewma_shrink_cov(returns_df, lam=0.94, shrink_lambda=0.0):
    T, N = returns_df.shape
    S = returns_df.cov().values
    cov_list = []
    for t in range(T):
        r = returns_df.iloc[t].values.reshape(-1, 1)
        S = lam * S + (1 - lam) * (r @ r.T)
        target = constant_correlation_target(S)
        S_shrink = shrink_lambda * target + (1 - shrink_lambda) * S
        cov_list.append(pd.DataFrame(S_shrink,
                                     index=returns_df.columns,
                                     columns=returns_df.columns))
    return cov_list

def optimize_weights(mu, cov, objective='sharpe', ridge=1e-3):
    mu_arr = mu.values
    cov_mat = cov.values
    N = len(mu_arr)

    if objective == 'sharpe':
        def obj(w):
            ret = w @ mu_arr
            vol = np.sqrt(w @ cov_mat @ w)
            ratio = -ret / vol if vol > 0 else np.inf
            penalty = ridge * np.sum(w ** 2)
            return ratio + penalty

    elif objective == 'kelly':
        def obj(w):
            utility = -(w @ mu_arr - 0.5 * w @ cov_mat @ w)
            penalty = ridge * np.sum(w ** 2)
            return utility + penalty

    else:
        raise ValueError("objective는 'sharpe' 또는 'kelly'만 가능합니다.")

    bounds = [(0, 1)] * N
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    w0 = np.ones(N) / N

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x if res.success else np.full(N, np.nan), index=mu.index)

def rolling_portfolio_weights(mu_df, cov_list, objective='sharpe', ridge=1e-3):
    weights = []
    for t in range(len(mu_df)):
        weights.append(optimize_weights(mu_df.iloc[t], cov_list[t], objective=objective, ridge=ridge))
    return pd.DataFrame(weights, index=mu_df.index)

# ===========================
# 실행 예시
# ===========================
tickers = ['133690.KS', '101280.KS', '279530.KS', '402970.KS', '442320.KS', '453870.KS' ,'473640.KS']
#            나스닥       토픽스          고배당       다우존스    원자력       니프티        금채굴
#tickers = ['QQQ','GLD']

prices = download_prices(tickers, '2005-01-01', '2025-12-31')
log_returns = compute_log_returns(prices)

mu_df, var_df, _ = estimate_dynamic_mu(log_returns, ar=1)
mu_shrunk = shrink_mean(mu_df, var_df=var_df, method='manual',lam= 0.2)

cov_series = ewma_shrink_cov(log_returns, lam=0.94, shrink_lambda=0.0)

sharpe_weights = rolling_portfolio_weights(mu_shrunk, cov_series, objective='sharpe', ridge= 0.15)
kelly_weights = rolling_portfolio_weights(mu_shrunk, cov_series, objective='kelly', ridge= 0.009)



import bt
from bt.algos import Or, RunOnce, RunIfOutOfBounds

# 1. 날짜 정렬 일치
common_idx = prices.index.intersection(kelly_weights.index)
prices_bt = prices.loc[common_idx]
weights_bt = kelly_weights.loc[common_idx]

# 전략 백테스트
strategy_test = bt.Strategy(
    'My Sharpe Strategy',
    [
        bt.algos.WeighTarget(weights_bt),
        Or([
            bt.algos.RunOnce(),
            bt.algos.RunIfOutOfBounds(0.1)
        ]),
        bt.algos.Rebalance()
    ]
)
test_bt = bt.Backtest(strategy_test, prices_bt)

# 1. SPY 데이터 다운로드 (Series → DataFrame 변환)
benchmark_price = bt.get('101280.KS', start=prices_bt.index[0], end=prices_bt.index[-1])
# 2. 날짜 정렬 일치
common_idx = prices_bt.index.intersection(benchmark_price.index)
prices_bt = prices_bt.loc[common_idx]
benchmark_price = benchmark_price.loc[common_idx]
weights_bt = weights_bt.loc[common_idx]

# 3. 전략 정의 (이미 test_bt 있음)
benchmark_bt = bt.Backtest(
    bt.Strategy('SPY',
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance()
        ]
    ),
    benchmark_price
)

# 4. 실행 및 비교
result = bt.run(test_bt, benchmark_bt)
result.display()
result.plot()
