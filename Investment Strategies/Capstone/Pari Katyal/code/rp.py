##RISK PARITY CODE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm

def load_data(file_path, sheet_name):
    """Load data from the specified Excel sheet."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data['x'] = pd.to_datetime(data['x'])
    data.set_index('x', inplace=True)
    return data

def calculate_daily_returns(data):
    """Calculate daily returns for each index."""
    returns = data.pct_change().dropna()
    return returns

def calculate_risk_estimates(returns):
    """Calculate the standard deviation of the daily returns for each index."""
    risk_estimates = returns.std()
    return risk_estimates

def calculate_risk_parity_allocations(risk_estimates):
    """Calculate risk parity allocations based on risk estimates."""
    inverse_risk = 1 / risk_estimates
    allocations = inverse_risk / inverse_risk.sum()
    allocations_percentage = allocations * 100
    return allocations_percentage

def rebalancing(returns, start_date, end_date, frequency='Q'):
    """Rebalance the portfolio based on the given frequency using risk parity allocations."""
    rebalancing_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    allocations_history = {}

    for date in rebalancing_dates:
        period_returns = returns.loc[:date]
        risk_estimates = calculate_risk_estimates(period_returns)
        allocations = calculate_risk_parity_allocations(risk_estimates)
        allocations_history[date] = allocations

    return pd.DataFrame(allocations_history).T

def calculate_rolling_correlations(returns, window=30):
    """Calculate rolling correlations for the returns with a specified window."""
    rolling_correlations = returns.rolling(window=window).corr()
    return rolling_correlations

def calculate_equally_weighted_allocations(returns):
    """Calculate equally weighted allocations."""
    num_assets = returns.shape[1]
    equal_weight = 100 / num_assets
    allocations = pd.Series([equal_weight] * num_assets, index=returns.columns)
    return allocations

def calculate_portfolio_returns(returns, allocations):
    """Calculate the portfolio returns based on allocations."""
    weighted_returns = returns.multiply(allocations, axis=1) / 100
    portfolio_returns = weighted_returns.sum(axis=1)
    return portfolio_returns

def calculate_performance_metrics(portfolio_returns):
    """Calculate performance metrics for the portfolio."""
    # Annualized return
    annualized_return = np.mean(portfolio_returns) * 252
    # Annualized volatility
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    # Sharpe ratio
    sharpe_ratio = annualized_return / annualized_volatility
    return annualized_return, annualized_volatility, sharpe_ratio

def calculate_max_drawdown(portfolio_returns):
    """Calculate the maximum drawdown for the portfolio."""
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_alpha_beta(portfolio_returns, benchmark_returns):
    """Calculate alpha and beta of the portfolio relative to a benchmark."""
    returns_df = pd.DataFrame({
        'Portfolio_Return': portfolio_returns,
        'Benchmark_Return': benchmark_returns
    }).dropna()

    # Add a constant for the regression intercept
    returns_df = sm.add_constant(returns_df)

    # Perform the regression to calculate alpha and beta
    model = sm.OLS(returns_df['Portfolio_Return'], returns_df[['const', 'Benchmark_Return']])
    results = model.fit()

    # Extract alpha and beta
    alpha = results.params['const']
    beta = results.params['Benchmark_Return']

    return alpha, beta

def plot_cumulative_returns(risk_parity_returns, equal_weight_returns):
    """Plot cumulative returns for both portfolios."""
    cum_returns_risk_parity = (1 + risk_parity_returns).cumprod()
    cum_returns_equal_weight = (1 + equal_weight_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(cum_returns_risk_parity, label='Risk Parity Portfolio', color = 'royalblue')
    plt.plot(cum_returns_equal_weight, label='Equally Weighted Portfolio', color = 'navy')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.show()

def subset_data_by_months(data, start_month, end_month):
    """Subset the data based on the given start and end month."""
    subset = data[(data.index >= start_month) & (data.index <= end_month)]
    return subset

def fit_garch_model(returns, p=1, q=1):
    """Fit a GARCH model to the returns and return the conditional volatility."""
    # Rescale returns
    scaled_returns = returns * 100
    garch_model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    conditional_volatility = garch_fit.conditional_volatility / 100  # Scale back the volatility
    return conditional_volatility

# Example usage
file_path = '/Users/parikatyal/Downloads/tes.xlsx'
sheet_name = 'Cumulative_Total_Returns_USD'
data = load_data(file_path, sheet_name)

# Subset data by months for lookback period testing
start_month = '2017-01-01'
end_month = '2020-01-01'
subset_data = subset_data_by_months(data, start_month, end_month)
returns = calculate_daily_returns(subset_data)

# Fit GARCH model to each index return series with different p and q values
garch_volatilities = returns.apply(fit_garch_model, p=1, q=1)  # You can change p and q values here

# Rebalancing from the start to the end of the subset data
start_date = returns.index.min()
end_date = returns.index.max()
allocations_history = rebalancing(returns, start_date, end_date, frequency='QE')  # You can change frequency here

# Calculate rolling correlations with a 60-day window
rolling_corrs = calculate_rolling_correlations(returns, window=60)

# Calculate equally weighted allocations
equal_weight_allocations = calculate_equally_weighted_allocations(returns)

# Calculate portfolio returns
risk_parity_returns = calculate_portfolio_returns(returns, allocations_history.iloc[-1])
equal_weight_returns = calculate_portfolio_returns(returns, equal_weight_allocations)

# Calculate performance metrics
risk_parity_metrics = calculate_performance_metrics(risk_parity_returns)
equal_weight_metrics = calculate_performance_metrics(equal_weight_returns)

# Calculate maximum drawdown
max_drawdown_risk_parity = calculate_max_drawdown(risk_parity_returns)
max_drawdown_equal_weight = calculate_max_drawdown(equal_weight_returns)

# Calculate alpha and beta
alpha, beta = calculate_alpha_beta(risk_parity_returns, equal_weight_returns)

# Print performance metrics
print("Risk Parity Portfolio Performance Metrics:")
print(f"Weights: {allocations_history.iloc[-1]}")
print(f"Annualized Return: {risk_parity_metrics[0]:.2%}")
print(f"Annualized Volatility: {risk_parity_metrics[1]:.2%}")
print(f"Sharpe Ratio: {risk_parity_metrics[2]:.2f}")
print(f"Maximum Drawdown: {max_drawdown_risk_parity:.2%}")
print(f"Alpha: {alpha:.2%}")
print(f"Beta: {beta:.2f}")

print("\nEqually Weighted Portfolio Performance Metrics:")
print(f"Annualized Return: {equal_weight_metrics[0]:.2%}")
print(f"Annualized Volatility: {equal_weight_metrics[1]:.2%}")
print(f"Sharpe Ratio: {equal_weight_metrics[2]:.2f}")
print(f"Maximum Drawdown: {max_drawdown_equal_weight:.2%}")

# Plot cumulative returns
plot_cumulative_returns(risk_parity_returns, equal_weight_returns)

# Plot GARCH volatilities for one of the indices as an example
plt.figure(figsize=(10, 6))
plt.plot(garch_volatilities['SPX Index'], label='GARCH Volatility - SPX Index')
plt.title('GARCH Conditional Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()
