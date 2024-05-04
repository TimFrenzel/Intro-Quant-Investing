import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
import matplotlib.pyplot as plt

# Fetch data for given tickers within a specified date range
def fetch_data(tickers, start_date='2014-01-01', end_date='2024-04-28'):
    data = yf.download(tickers, start=start_date, end=end_date)
    adjusted_closes = data['Adj Close'].dropna(how='all', axis=1).ffill()  
    return adjusted_closes

# Calculate percentage returns from the adjusted closing prices
def calculate_returns(data):
    returns = data.pct_change().dropna() * 100  
    return returns

def calculate_beta(portfolio_returns, benchmark_returns):
    covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

# Calculate expected returns based on historical data
def calculate_expected_returns(returns):
    return returns.mean()  # Simple mean of historical returns

# Calculate the dynamic correlation over a rolling window with a minimum number of periods
def calculate_dynamic_correlation(returns, window_size=60):
    min_periods = int(window_size * 0.8)  # Require at least 80% of the window size
    rolling_corr = returns.rolling(window=window_size, min_periods=min_periods).corr()  
    return rolling_corr

# Calculate asymmetric volatility using GARCH model
def calculate_asymmetric_volatility(returns):
    volatilities = {}
    for asset in returns.columns:
        model = arch_model(returns[asset].dropna(), vol='Garch', p=1, o=1, q=1, mean='Zero', rescale=False)
        fitted_model = model.fit(disp='off')
        volatilities[asset] = fitted_model.conditional_volatility
    return pd.DataFrame(volatilities)

# Adjusted risk parity calculation incorporating expected returns
def calculate_adjusted_risk_parity(volatility, correlation, expected_returns, alpha= .94):
    if not np.isfinite(correlation.values).all() or correlation.shape[0] != correlation.shape[1]:
        correlation = pd.DataFrame(np.eye(len(volatility)), index=volatility.index, columns=volatility.index)
    inv_volatility = 1 / volatility
    inverse_corr = np.linalg.inv(correlation.values)
    raw_weights = np.dot(inverse_corr, inv_volatility)
    risk_parity_weights = raw_weights / raw_weights.sum()
    expected_returns_scaled = expected_returns / expected_returns.max()
    combined_weights = alpha * risk_parity_weights + (1 - alpha) * expected_returns_scaled
    combined_weights /= combined_weights.sum()
    return pd.Series(combined_weights, index=volatility.index)
def integrate_6040_portfolio(data):
    weights = pd.Series(index=data.columns, data=0)
    if 'SPY' in data.columns and 'AGG' in data.columns:
        weights['SPY'] = 0.6
        weights['AGG'] = 0.4
    else:
        weights.fill(1 / len(data.columns))
    return weights

# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.00):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    return sharpe_ratio

# Function to calculate the maximum drawdown
def calculate_max_drawdown(cumulative_returns):
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

# Subset data based on a lookback period in months
def subset_data(data, lookback_months):
    end_date = data.index.max()
    start_date = end_date - pd.DateOffset(months=lookback_months)
    return data.loc[start_date:end_date]

def plot_cumulative_returns(portfolio_returns_6040, portfolio_returns_rp, start_date, end_date):
    # Calculate cumulative returns
    cumulative_returns_6040 = (1 + portfolio_returns_6040 / 100).cumprod()
    cumulative_returns_rp = (1 + portfolio_returns_rp / 100).cumprod()
    
    # Plotting the cumulative returns
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns_6040.index, cumulative_returns_6040, label='60/40 Portfolio')
    plt.plot(cumulative_returns_rp.index, cumulative_returns_rp, label='Risk Parity Portfolio',  color='blue'),  
    
    plt.title('Cumulative Returns of both Portfolios')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

# Integration of the Risk Parity strategy into the backtesting framework
def integrate_and_backtest_full_strategy(tickers, benchmark_ticker='SPY', start_date='2014-01-01', end_date='2024-04-28', lookback_months=120):
    # Fetch the data for both the portfolios and the benchmark
    all_tickers = tickers + [benchmark_ticker]
    data = fetch_data(all_tickers, start_date, end_date)
    
    # Subset data based on lookback period
    data_subset = subset_data(data, lookback_months)
    
    # Calculate returns for the subset
    returns = calculate_returns(data_subset)
    
    # Benchmark returns are calculated here once
    benchmark_returns = returns[benchmark_ticker]

    # Risk Parity Portfolio Calculations
    expected_returns = calculate_expected_returns(returns)
    dynamic_correlation = calculate_dynamic_correlation(returns)
    last_correlation = dynamic_correlation.groupby(level=0).tail(1).droplevel(level=0).dropna(axis=1, how='any')
    volatility = calculate_asymmetric_volatility(returns)
    weights_rp = calculate_adjusted_risk_parity(volatility.iloc[-1], last_correlation, expected_returns)
    portfolio_returns_rp = (returns * weights_rp).sum(axis=1)
    beta_rp = calculate_beta(portfolio_returns_rp, benchmark_returns)
    cumulative_returns_rp = (1 + portfolio_returns_rp / 100).cumprod() - 1
    sharpe_ratio_rp = calculate_sharpe_ratio(portfolio_returns_rp)
    max_drawdown_rp = calculate_max_drawdown(cumulative_returns_rp)
    annualized_volatility_rp = portfolio_returns_rp.std() * np.sqrt(252)
   
    # 60/40 Portfolio Calculations
    weights_6040 = integrate_6040_portfolio(data_subset)
    portfolio_returns_6040 = (returns * weights_6040).sum(axis=1)
    cumulative_returns_6040 = (1 + portfolio_returns_6040 / 100).cumprod() - 1
    sharpe_ratio_6040 = calculate_sharpe_ratio(portfolio_returns_6040)
    max_drawdown_6040 = calculate_max_drawdown(cumulative_returns_6040)
    annualized_volatility_6040 = portfolio_returns_6040.std() * np.sqrt(252)
    beta_6040 = calculate_beta(portfolio_returns_6040, benchmark_returns)
    
    plot_cumulative_returns(portfolio_returns_6040, portfolio_returns_rp, start_date, end_date)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # Performance comparison in DataFrame
    performance_comparison = pd.DataFrame({
        'Strategy': ['Risk Parity', '60/40 SPY/AGG'],
        'Cumulative Return': [cumulative_returns_rp.iloc[-1], cumulative_returns_6040.iloc[-1]],
        'Volatility': [annualized_volatility_rp, annualized_volatility_6040],
        'Sharpe Ratio': [sharpe_ratio_rp, sharpe_ratio_6040],
        'Maximum Drawdown': [max_drawdown_rp, max_drawdown_6040],
        'Beta': [beta_rp, beta_6040],
    })
    print("Risk Parity Weights:", weights_rp)
    print("60/40 Weights:", weights_6040)
    print(performance_comparison)



#tickers for risk parity
tickers = ['SPY', 'EFA', 'VWO', 'TLT', 'HYG', 'BNDX', 'NVDA', 'GLD', 'DBC', 'IGF', 'PSP', 'AGG']

#run strategy
integrate_and_backtest_full_strategy(tickers, start_date='2014-01-01', end_date='2024-04-28', lookback_months= 120)

