import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from arch import arch_model
import statsmodels.api as sm
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(file_path, sheet_name):
    """Load data from the specified Excel sheet."""
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data['x'] = pd.to_datetime(data['x'])
    data.set_index('x', inplace=True)
    return data

def calculate_daily_returns(data):
    """Calculate daily returns for each index."""
    returns = data.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    return returns

def estimate_mean_covariance(returns):
    """Estimate the mean returns and covariance matrix."""
    mean_returns = returns.mean() * 252  # Annualize mean returns
    covariance_matrix = returns.cov() * 252  # Annualize covariance matrix
    return mean_returns, covariance_matrix

def portfolio_performance(weights, mean_returns, covariance_matrix):
    """Calculate portfolio performance metrics: return and volatility."""
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate=0.0):
    """Calculate the negative Sharpe ratio for given weights."""
    p_returns, p_volatility = portfolio_performance(weights, mean_returns, covariance_matrix)
    sharpe_ratio = (p_returns - risk_free_rate) / p_volatility
    return -sharpe_ratio  # Minimize negative Sharpe ratio

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
    
    return result.x

def calculate_max_drawdown(portfolio_returns):
    """Calculate the maximum drawdown for the portfolio."""
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_performance_metrics(portfolio_returns, benchmark_returns):
    """Calculate performance metrics for the portfolio."""
    # Align the indices of the returns
    aligned_returns = portfolio_returns.align(benchmark_returns, join='inner')
    portfolio_returns = aligned_returns[0]
    benchmark_returns = aligned_returns[1]
    
    # Annualized return
    annualized_return = np.mean(portfolio_returns) * 252
    # Annualized volatility
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    # Sharpe ratio
    sharpe_ratio = annualized_return / annualized_volatility
    # Maximum drawdown
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    return annualized_return, annualized_volatility, sharpe_ratio, max_drawdown

def calculate_portfolio_returns(returns, allocations):
    """Calculate the portfolio returns based on allocations."""
    weighted_returns = returns.multiply(allocations, axis=1)
    portfolio_returns = weighted_returns.sum(axis=1)
    return portfolio_returns

def rebalance_portfolio(returns, weights, rebalancing_dates):
    """Rebalance the portfolio at the end of each specified period."""
    rebalanced_returns = []
    dates = returns.index
    current_weights = weights.copy()
    
    for i in range(len(dates)):
        if dates[i] in rebalancing_dates:
            # Rebalance at the specified dates
            current_weights = weights.copy()
        
        # Calculate portfolio return for the current day
        daily_return = np.dot(current_weights, returns.iloc[i])
        rebalanced_returns.append(daily_return)
    
    return pd.Series(rebalanced_returns, index=dates)

def equally_weighted_portfolio(returns):
    """Calculate returns for an equally weighted portfolio."""
    num_assets = returns.shape[1]
    equal_weights = np.ones(num_assets) / num_assets
    equal_weighted_returns = calculate_portfolio_returns(returns, equal_weights)
    return equal_weighted_returns, equal_weights

def calculate_benchmark_returns(benchmark_returns):
    """Calculate cumulative returns for the benchmark."""
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    return benchmark_cum_returns

def subset_data_by_months(data, start_month, end_month):
    """Subset the data based on the given start and end month."""
    subset = data[(data.index >= start_month) & (data.index <= end_month)]
    return subset

def fit_garch_model(returns, p=1, q=1):
    """Fit a GARCH model to the returns and return the conditional volatility."""
    scaled_returns = returns * 100
    garch_model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            garch_fit = garch_model.fit(disp="off")
        except Exception as e:
            print(f"GARCH model fitting failed for {returns.name} with error: {e}")
            return pd.Series([np.nan] * len(returns), index=returns.index)
    conditional_volatility = garch_fit.conditional_volatility / 100
    return conditional_volatility

def estimate_garch_volatility(returns):
    """Estimate annualized volatility using GARCH(1,1) model."""
    volatilities = {}
    for column in returns.columns:
        scaled_returns = returns[column] * 100  # Rescale to avoid poor scaling issues
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp="off")
        # Extract the last forecasted variance
        forecasted_variance = model_fit.conditional_volatility.iloc[-1] ** 2
        # Annualize the volatility
        annualized_volatility = np.sqrt(forecasted_variance * 252) / 100  # Scale back
        volatilities[column] = annualized_volatility
    return volatilities

def calculate_garch_covariance(returns, garch_volatilities):
    """Calculate the covariance matrix using GARCH volatilities and historical correlations."""
    corr_matrix = returns.corr()
    vol_vector = np.array([garch_volatilities[col] for col in returns.columns])
    cov_matrix = corr_matrix * np.outer(vol_vector, vol_vector)
    return cov_matrix

def calculate_alpha_beta(portfolio_returns, benchmark_returns):
    """Calculate alpha and beta of the portfolio relative to a benchmark."""
    aligned_returns = portfolio_returns.align(benchmark_returns, join='inner')
    portfolio_returns = aligned_returns[0]
    benchmark_returns = aligned_returns[1]
    
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

def walk_forward_backtest(data, initial_train_period, test_period, rebalancing_freq='QE', strategy='rp', use_garch=False):
    """Perform walk-forward backtest on the given data."""
    start_date = data.index[0]
    end_date = data.index[-1]
    
    current_date = start_date + initial_train_period
    
    portfolio_returns = pd.Series(dtype=float)
    rebalancing_dates = pd.date_range(start=current_date, end=end_date, freq=rebalancing_freq)
    
    while current_date + test_period <= end_date:
        train_data = data[:current_date]
        test_data = data[current_date:current_date + test_period]
        
        returns = calculate_daily_returns(train_data)
        
        if strategy == 'rp':
            if use_garch:
                garch_volatilities = estimate_garch_volatility(returns)
                covariance_matrix = calculate_garch_covariance(returns, garch_volatilities)
            else:
                _, covariance_matrix = estimate_mean_covariance(returns)
            num_assets = len(covariance_matrix)
            initial_weights = np.ones(num_assets) / num_assets
            constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            result = minimize(risk_parity_objective, initial_weights, args=covariance_matrix, method='SLSQP', bounds=bounds, constraints=constraints)
            allocations = result.x
        
        elif strategy == 'mvo':
            if use_garch:
                garch_volatilities = estimate_garch_volatility(returns)
                covariance_matrix = calculate_garch_covariance(returns, garch_volatilities)
                mean_returns = returns.mean() * 252  # Ensure mean_returns is defined
            else:
                mean_returns, covariance_matrix = estimate_mean_covariance(returns)
            allocations = maximize_sharpe_ratio_with_constraints(mean_returns, covariance_matrix)
        
        test_returns = calculate_daily_returns(test_data)
        portfolio_return = rebalance_portfolio(test_returns, allocations, rebalancing_dates)
        
        if not portfolio_return.empty:
            portfolio_returns = pd.concat([portfolio_returns, portfolio_return])
        
        current_date += test_period
    
    portfolio_returns = portfolio_returns.sort_index()
    return portfolio_returns

def risk_parity_objective(weights, covariance_matrix):
    """Objective function for risk parity optimization."""
    risk_contributions = calculate_risk_contributions(weights, covariance_matrix)
    mean_contribution = np.mean(risk_contributions)
    objective_value = np.sum((risk_contributions - mean_contribution) ** 2)
    return objective_value

def calculate_risk_contributions(weights, covariance_matrix):
    """Calculate the risk contributions of each asset."""
    portfolio_variance = weights.T @ covariance_matrix @ weights
    marginal_contributions = covariance_matrix @ weights
    risk_contributions = marginal_contributions * weights / np.sqrt(portfolio_variance)
    return risk_contributions

# Load the provided data
file_path = '/Users/parikatyal/Downloads/CapData.xlsx'
sheet_name = 'Cumulative_Total_Returns_USD'
data = load_data(file_path, sheet_name)

# Define walk-forward backtest parameters
initial_train_period = pd.DateOffset(years=1)
test_period = pd.DateOffset(years=1)

# Set backtest start and end dates
start_date = '2000-01-01'
end_date = '2020-01-01'
data = subset_data_by_months(data, start_date, end_date)

# Perform walk-forward backtest with Risk Parity strategy
rp_portfolio_returns = walk_forward_backtest(data, initial_train_period, test_period, strategy='rp', use_garch=True)

# Perform walk-forward backtest with MVO strategy
mvo_portfolio_returns = walk_forward_backtest(data, initial_train_period, test_period, strategy='mvo', use_garch=True)

# Calculate the equally weighted portfolio returns to use as a benchmark
equal_weighted_returns, equal_weights = equally_weighted_portfolio(calculate_daily_returns(data))

# Calculate performance metrics for the Risk Parity portfolio
rp_performance_metrics = calculate_performance_metrics(rp_portfolio_returns, equal_weighted_returns)
rp_max_drawdown = calculate_max_drawdown(rp_portfolio_returns)
rp_alpha, rp_beta = calculate_alpha_beta(rp_portfolio_returns, equal_weighted_returns)

# Calculate performance metrics for the MVO portfolio
mvo_performance_metrics = calculate_performance_metrics(mvo_portfolio_returns, equal_weighted_returns)
mvo_max_drawdown = calculate_max_drawdown(mvo_portfolio_returns)
mvo_alpha, mvo_beta = calculate_alpha_beta(mvo_portfolio_returns, equal_weighted_returns)

# Print performance metrics for Risk Parity-based backtest
print("Walk-Forward Backtest Performance Metrics with Risk Parity Strategy:")
print(f"Annualized Return: {rp_performance_metrics[0]:.2%}")
print(f"Annualized Volatility: {rp_performance_metrics[1]:.2%}")
print(f"Sharpe Ratio: {rp_performance_metrics[2]:.2f}")
print(f"Maximum Drawdown: {rp_max_drawdown:.2%}")
print(f"Beta: {rp_beta:.2f}")

# Print performance metrics for MVO-based backtest
print("\nWalk-Forward Backtest Performance Metrics with MVO Strategy:")
print(f"Annualized Return: {mvo_performance_metrics[0]:.2%}")
print(f"Annualized Volatility: {mvo_performance_metrics[1]:.2%}")
print(f"Sharpe Ratio: {mvo_performance_metrics[2]:.2f}")
print(f"Maximum Drawdown: {mvo_max_drawdown:.2%}")
print(f"Beta: {mvo_beta:.2f}")

# Print performance metrics for the equally weighted portfolio
equal_weighted_performance_metrics = calculate_performance_metrics(equal_weighted_returns, equal_weighted_returns)
equal_weighted_max_drawdown = calculate_max_drawdown(equal_weighted_returns)
equal_weighted_alpha, equal_weighted_beta = calculate_alpha_beta(equal_weighted_returns, equal_weighted_returns)

print("\nPerformance Metrics for Equally Weighted Portfolio:")
print(f"Annualized Return: {equal_weighted_performance_metrics[0]:.2%}")
print(f"Annualized Volatility: {equal_weighted_performance_metrics[1]:.2%}")
print(f"Sharpe Ratio: {equal_weighted_performance_metrics[2]:.2f}")
print(f"Maximum Drawdown: {equal_weighted_max_drawdown:.2%}")
print(f"Beta: {equal_weighted_beta:.2f}")

# Plot cumulative returns for comparison starting from 2005
cumulative_returns_start = '2005-01-01'
rp_cum_returns = (1 + rp_portfolio_returns[rp_portfolio_returns.index >= cumulative_returns_start]).cumprod()
mvo_cum_returns = (1 + mvo_portfolio_returns[mvo_portfolio_returns.index >= cumulative_returns_start]).cumprod()
eq_cum_returns = (1 + equal_weighted_returns[equal_weighted_returns.index >= cumulative_returns_start]).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(rp_cum_returns, label='Risk Parity Portfolio', color='royalblue')
plt.plot(mvo_cum_returns, label='MVO Portfolio', color='blue')
plt.plot(eq_cum_returns, label='Equally Weighted Portfolio', color='navy')
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Calculate and plot rolling alpha as a line plot
rolling_window = 252  # Rolling window of 1 year

# Calculate rolling alpha
rp_rolling_alpha = rp_portfolio_returns.rolling(window=rolling_window).apply(lambda x: calculate_alpha_beta(x, equal_weighted_returns)[0])
mvo_rolling_alpha = mvo_portfolio_returns.rolling(window=rolling_window).apply(lambda x: calculate_alpha_beta(x, equal_weighted_returns)[0])

# Remove NaN values from rolling alpha
rp_rolling_alpha = rp_rolling_alpha.dropna()
mvo_rolling_alpha = mvo_rolling_alpha.dropna()

# Plot rolling alpha as a line plot
plt.figure(figsize=(12, 6))
plt.plot(rp_rolling_alpha, label='Risk Parity Rolling Alpha', color='royalblue')
plt.plot(mvo_rolling_alpha, label='MVO Rolling Alpha', color='blue')
plt.title('Rolling Alpha (1-Year Window) Comparison')
plt.xlabel('Date')
plt.ylabel('Alpha')
plt.legend()
plt.show()
