import pandas as pd
import numpy as np
from scipy.optimize import minimize
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load data
def load_data(file_path, sheet_name):
    """Load data from the specified Excel sheet."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data['x'] = pd.to_datetime(data['x'])
    data.set_index('x', inplace=True)
    return data

# Calculate daily returns
def calculate_daily_returns(data):
    """Calculate daily returns for each index."""
    returns = data.pct_change().dropna()
    return returns

# Calculate GARCH Volatilities
def estimate_garch_volatility(returns):
    """Estimate annualized volatility using GARCH(1,1) model."""
    volatilities = {}
    for column in returns.columns:
        scaled_returns = returns[column] * 100
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp="off")
        forecasted_variance = model_fit.conditional_volatility ** 2
        annualized_volatility = np.sqrt(forecasted_variance * 252) / 100
        volatilities[column] = annualized_volatility.iloc[-1]
    return volatilities

# Clip extreme values
def clip_extreme_values(returns):
    """Clip extreme values in the returns data."""
    clipped_returns = returns.copy()
    for column in clipped_returns.columns:
        upper_limit = clipped_returns[column].mean() + 3 * clipped_returns[column].std()
        lower_limit = clipped_returns[column].mean() - 3 * clipped_returns[column].std()
        clipped_returns[column] = clipped_returns[column].clip(lower=lower_limit, upper=upper_limit)
    return clipped_returns

# Calculate GARCH Covariance Matrix
def calculate_garch_covariance(returns, garch_volatilities):
    """Calculate the covariance matrix using GARCH volatilities and historical correlations."""
    corr_matrix = returns.corr()
    vol_vector = np.array([garch_volatilities[col] for col in returns.columns])
    cov_matrix = corr_matrix * np.outer(vol_vector, vol_vector)
    return cov_matrix

# Calculate Performance Metrics
def calculate_performance_metrics(portfolio_returns):
    """Calculate performance metrics for the portfolio."""
    annualized_return = np.mean(portfolio_returns) * 252
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    return annualized_return, annualized_volatility, sharpe_ratio

# Calculate maximum drawdown
def calculate_max_drawdown(portfolio_returns):
    """Calculate the maximum drawdown for the portfolio."""
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

# Calculate alpha and beta
def calculate_alpha_beta(portfolio_returns, benchmark_returns):
    """Calculate alpha and beta of the portfolio relative to a benchmark."""
    returns_df = pd.DataFrame({
        'Portfolio_Return': portfolio_returns,
        'Benchmark_Return': benchmark_returns
    }).dropna()
    returns_df = sm.add_constant(returns_df)
    model = sm.OLS(returns_df['Portfolio_Return'], returns_df[['const', 'Benchmark_Return']])
    results = model.fit()
    alpha = results.params['const']
    beta = results.params['Benchmark_Return']
    return alpha, beta

# Calculate equally weighted allocations
def calculate_equally_weighted_allocations(returns):
    """Calculate equally weighted allocations."""
    num_assets = returns.shape[1]
    equal_weight = 1 / num_assets
    allocations = pd.Series([equal_weight] * num_assets, index=returns.columns)
    return allocations

# Calculate portfolio returns
def calculate_portfolio_returns(returns, allocations):
    """Calculate the portfolio returns based on allocations."""
    weighted_returns = returns.multiply(allocations, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    return portfolio_returns

# Subset data
def subset_data(data, start_date, end_date):
    """Subset the data based on the given date range."""
    subset = data[(data.index >= start_date) & (data.index <= end_date)]
    return subset

# Maximize Sharpe Ratio with Constraints
def maximize_sharpe_ratio_with_constraints(mean_returns, covariance_matrix, risk_free_rate=0.0, min_weight=0.01, max_weight=0.50):
    """Maximize the Sharpe ratio to find the optimal portfolio weights with weight constraints."""
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix, risk_free_rate)
    
    # Constraint: sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Bounds for individual weights
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed.")
    return result.x

# Calculate negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate=0.0):
    """Calculate the negative Sharpe ratio for given weights."""
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, covariance_matrix)
    sharpe_ratio = (p_returns - risk_free_rate) / p_volatility
    return -sharpe_ratio  # Minimize negative Sharpe ratio

# Portfolio performance
def portfolio_performance(weights, mean_returns, covariance_matrix):
    """Calculate portfolio performance metrics: return and volatility."""
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return returns, volatility

# Rebalance portfolio
def rebalance_portfolio(returns, weights):
    """Rebalance the portfolio at the end of each quarter."""
    rebalanced_returns = []
    dates = returns.index
    current_weights = weights.copy()
    
    for i in range(len(dates)):
        if i > 0 and (dates[i].quarter != dates[i-1].quarter or dates[i].year != dates[i-1].year):
            current_weights = weights.copy()
        daily_return = np.dot(current_weights, returns.iloc[i])
        rebalanced_returns.append(daily_return)
    
    return pd.Series(rebalanced_returns, index=dates)

# Example usage
file_path = '/Users/parikatyal/Downloads/CapData.xlsx'
sheet_name = 'Cumulative_Total_Returns_USD'
data = load_data(file_path, sheet_name)

# Calculate daily returns
returns = calculate_daily_returns(data)
print("Daily Returns:")
print(returns.head())  # Print the first few rows of daily returns for inspection

# Clean returns data
cleaned_returns = clip_extreme_values(returns)
print("Cleaned Returns:")
print(cleaned_returns.head())  # Print the first few rows of cleaned returns for inspection

# Subset data for lookback testing
start_date = '2017-01-01'
end_date = '2020-01-01'
subset_returns = subset_data(cleaned_returns, start_date, end_date)
print("Subset Returns:")
print(subset_returns.head())  # Print the first few rows of subset returns for inspection

# Calculate GARCH volatilities and covariance matrix
garch_volatilities = estimate_garch_volatility(subset_returns)
covariance_matrix = calculate_garch_covariance(subset_returns, garch_volatilities)

# Optimize for Risk Parity
inv_vol = {k: 1 / v for k, v in garch_volatilities.items()}
sum_inv_vol = sum(inv_vol.values())
initial_weights = {k: v / sum_inv_vol for k, v in inv_vol.items()}
weights_series = pd.Series(initial_weights)

# Normalize the weights to ensure they sum to 1
normalized_weights = weights_series / weights_series.sum()

# Calculate portfolio returns for the Risk Parity Portfolio
risk_parity_returns = calculate_portfolio_returns(subset_returns, normalized_weights)

# Calculate performance metrics for the Risk Parity Portfolio
risk_parity_metrics = calculate_performance_metrics(risk_parity_returns)
print("Risk Parity Portfolio Performance Metrics:")
print(f"Annualized Return: {risk_parity_metrics[0]:.2%}")
print(f"Annualized Volatility: {risk_parity_metrics[1]:.2%}")
print(f"Sharpe Ratio: {risk_parity_metrics[2]:.2f}")

# Equally Weighted Portfolio
equal_weight_allocations = calculate_equally_weighted_allocations(subset_returns)
equal_weight_returns = calculate_portfolio_returns(subset_returns, equal_weight_allocations)
print("Equally Weighted Portfolio Returns:")
print(equal_weight_returns.head(10))

equal_weight_metrics = calculate_performance_metrics(equal_weight_returns)
print("\nEqually Weighted Portfolio Performance Metrics:")
print(f"Annualized Return: {equal_weight_metrics[0]:.2%}")
print(f"Annualized Volatility: {equal_weight_metrics[1]:.2%}")
print(f"Sharpe Ratio: {equal_weight_metrics[2]:.2f}")

# Calculate maximum drawdown
# Calculate maximum drawdown for equally weighted portfolio
max_drawdown_equal_weight = calculate_max_drawdown(equal_weight_returns)
print(f"Maximum Drawdown: {max_drawdown_equal_weight:.2%}")

# MVO
mean_returns_subset = subset_returns.mean() * 252
optimal_weights_subset = maximize_sharpe_ratio_with_constraints(mean_returns_subset, covariance_matrix)

# Print the optimal weights for the MVO portfolio
print("Optimal Weights for MVO Portfolio (Subset Data with GARCH):")
print(optimal_weights_subset)

# Calculate performance metrics with quarterly rebalancing for the subset data
mvo_portfolio_returns_subset = rebalance_portfolio(subset_returns, optimal_weights_subset)
print("MVO Portfolio Returns:")
print(mvo_portfolio_returns_subset.head(10))

# Print performance metrics for the MVO portfolio
mvo_annualized_return, mvo_annualized_volatility, mvo_sharpe_ratio = calculate_performance_metrics(mvo_portfolio_returns_subset)
mvo_max_drawdown = calculate_max_drawdown(mvo_portfolio_returns_subset)
mvo_alpha, mvo_beta = calculate_alpha_beta(mvo_portfolio_returns_subset, equal_weight_returns)

print("\nPerformance Metrics for MVO Portfolio (Subset Data with GARCH):")
print(f"Annualized Return: {mvo_annualized_return:.2%}")
print(f"Annualized Volatility: {mvo_annualized_volatility:.2%}")
print(f"Sharpe Ratio: {mvo_sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {mvo_max_drawdown:.2%}")
print(f"Alpha: {mvo_alpha:.2f}")
print(f"Beta: {mvo_beta:.2f}")

# Calculate cumulative returns for all portfolios for the subset data
cum_returns_risk_parity = (1 + risk_parity_returns).cumprod() - 1
cum_returns_equal_weighted_subset = (1 + equal_weight_returns).cumprod() - 1
cum_returns_mvo_subset = (1 + mvo_portfolio_returns_subset).cumprod() - 1

# Plot cumulative returns for comparison for the subset data
plt.figure(figsize=(10, 6))
plt.plot(cum_returns_risk_parity, label='Risk Parity Portfolio', color='royalblue')
plt.plot(cum_returns_equal_weighted_subset, label='Equally Weighted Portfolio', color='navy')
plt.plot(cum_returns_mvo_subset, label='MVO Portfolio', color='blue')
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
