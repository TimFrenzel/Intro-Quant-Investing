library(quantmod)
library(dplyr)
library(PerformanceAnalytics)
library(zoo)  # for rollapply

# Function to implement the momentum strategy
momentum_strategy <- function(stock1_data, stock2_data, lookback_period, allocation_percent_change) {
  
  stock1 <- stock1_data
  stock2 <- stock2_data
  
  # Monthly returns calculation
  monthly_returns1 <- monthlyReturn(stock1)
  monthly_returns2 <- monthlyReturn(stock2)
  
  # Calculate positive months within the lookback period
  pos_months1 <- rollapply(monthly_returns1, width = lookback_period, FUN = function(x) sum(x > 0), by.column = TRUE, align = "right", fill = NA)
  pos_months2 <- rollapply(monthly_returns2, width = lookback_period, FUN = function(x) sum(x > 0), by.column = TRUE, align = "right", fill = NA)
  
  # Initialize allocations
  allocation1 <- rep(0.5, length(monthly_returns1))
  allocation2 <- rep(0.5, length(monthly_returns2))
  
  # Adjust allocations only at rebalancing points
  for (i in seq_along(allocation1)) {
    if (i > 1) {  # Ensure the first index doesn't use non-existing index 0
      if (i %% lookback_period == 0 && !is.na(pos_months1[i]) && !is.na(pos_months2[i])) {
        if (pos_months1[i] > pos_months2[i]) {
          # Increase allocation for stock1 by a percentage of its last allocation
          allocation1[i] <- min(1, allocation1[i-1] + allocation_percent_change/100)
          allocation2[i] <- max(0.05, allocation2[i-1] - allocation_percent_change/100)
        } else if (pos_months1[i] < pos_months2[i]) {
          allocation1[i] <- max(0.05, allocation1[i-1] - allocation_percent_change/100)
          allocation2[i] <- min(1, allocation2[i-1] + allocation_percent_change/100)
        } else {
          # Keep the current allocations if the number of positive months is the same
          allocation1[i] <- allocation1[i-1]
          allocation2[i] <- allocation2[i-1]
        }
      } else {
        allocation1[i] <- allocation1[i-1]
        allocation2[i] <- allocation2[i-1]
      }
    }
  }
  
  # Normalize to ensure allocations sum to 100%
  total_allocations <- allocation1 + allocation2
  allocation1 <- allocation1 / total_allocations
  allocation2 <- allocation2 / total_allocations
  
  # Prepare results data frame
  results <- data.frame(
    Date = index(monthly_returns1),
    Stock1_Returns = monthly_returns1,
    Stock2_Returns = monthly_returns2,
    Stock1_Allocation = allocation1,
    Stock2_Allocation = allocation2
  )
  
  results$Portfolio_Returns <- with(results, Stock1_Allocation * monthly.returns + Stock2_Allocation * monthly.returns.1)
  
  # Optional: Calculate cumulative returns if desired
  results$Cumulative_Returns <- cumprod(1 + results$Portfolio_Returns) - 1
  
  return(results)
}


k_fold_backtest <- function(stock1_data, stock2_data, lookback_period, allocation_percent_change, k) {
  # Calculate the number of observations per fold
  n <- nrow(stock1_data)
  fold_size <- floor(n / k)
  
  # Initialize results storage
  all_results <- data.frame()
  performance_metrics <- data.frame(Fold = integer(), SharpeRatio = numeric(), MaxDrawdown = numeric(), AnnualReturn = numeric())
  
  for (i in 1:k) {
    # Define the training and testing indices
    test_indices <- ((i - 1) * fold_size + 1):(i * fold_size)
    if (i == k) {
      test_indices <- ((i - 1) * fold_size + 1):n
    }
    train_indices <- setdiff(1:n, test_indices)
    
    # Split the data
    train_stock1_data <- stock1_data[train_indices]
    train_stock2_data <- stock2_data[train_indices]
    test_stock1_data <- stock1_data[test_indices]
    test_stock2_data <- stock2_data[test_indices]
    
    # Run the strategy on the training data
    train_results <- momentum_strategy(train_stock1_data, train_stock2_data, lookback_period, allocation_percent_change)
    
    # Run the strategy on the test data
    test_results <- momentum_strategy(test_stock1_data, test_stock2_data, lookback_period, allocation_percent_change)
    
    # Ensure the Portfolio_Returns are formatted as an xts object
    test_returns_xts <- xts(test_results$Portfolio_Returns, order.by = as.Date(test_results$Date))
    
    # Calculate performance metrics for the test period
    sharpe_ratio <- SharpeRatio(test_returns_xts, Rf = 0, p = 0.95, FUN = "StdDev")[1,1]
    # max_drawdown <- maxDrawdown(test_returns_xts)$maxDrawdown
    annual_return <- Return.annualized(test_returns_xts, geometric = TRUE)[1,1]
    
    # Append performance metrics
    performance_metrics <- rbind(performance_metrics, data.frame(Fold = i, SharpeRatio = sharpe_ratio, AnnualReturn = annual_return))
    
    # Append the test results to the all_results dataframe
    all_results <- rbind(all_results, test_results)
  }
  
  return(list(results = all_results, performance_metrics = performance_metrics))
}

# Backtesting the strategy with k-fold cross-validation
start_date <- as.Date("2005-01-01")
end_date <- as.Date("2020-05-03")
stock1_data <- getSymbols("SPY", src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
stock2_data <- getSymbols("AGG", src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)

# Number of folds
k <- 5

backtest <- k_fold_backtest(stock1_data, stock2_data, 12, 5, k)
results <- backtest$results
performance_metrics <- backtest$performance_metrics

# Plot the cumulative returns
ggplot(results, aes(x = Date)) +
  geom_line(aes(y = Cumulative_Returns, color = "Cumulative Returns")) +
  labs(title = "Momentum Strategy K-Fold Backtest, 12 month, 5%",
       x = "Date",
       y = "Cumulative Returns") +
  theme_minimal()

# Display performance metrics
print(performance_metrics)

