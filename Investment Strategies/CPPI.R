cppi_strategy <- function(risky_returns, safe_returns, floor, multiplier, dates) {
  initial_value <- 100
  n <- length(risky_returns)
  portfolio_value <- rep(initial_value, n)
  cushion <- rep(0, n)
  allocation_risky <- rep(0, n)
  allocation_safe <- rep(0, n)
  
  # Convert dates to Year to identify when the year changes
  years <- format(dates, "%Y")
  
  # Ensure there are no NA years; if there are, handle or exit function
  if (any(is.na(years))) {
    stop("NA values found in year calculations. Check the 'dates' vector.")
  }
  
  # Initialize a list to keep track of floor values by year
  floor_values <- setNames(rep(floor * initial_value, length(unique(years))), unique(years))
  
  for (t in 1:(n-1)) {
    # Check if the year has changed and update floor value
    if (t < n && years[t + 1] != years[t]) {
      floor_values[years[t + 1]] <- max(portfolio_value[t] * floor, floor_values[years[t]])
    }
    
    cushion[t] <- max(portfolio_value[t] - floor_values[years[t]], 0)
    allocation_risky[t] <- min(cushion[t] * multiplier, portfolio_value[t])
    allocation_safe[t] <- portfolio_value[t] - allocation_risky[t]
    portfolio_value[t + 1] <- allocation_risky[t] * (1 + risky_returns[t]) + allocation_safe[t] * (1 + safe_returns[t])
  }
  
  portfolio_returns <- c(NA, diff(portfolio_value) / portfolio_value[-n])
  return(list(portfolio_returns = portfolio_returns,
              allocation_risky = allocation_risky,
              allocation_safe = allocation_safe,
              floor_values = floor_values))
}

# Example usage with hypothetical returns and parameters
risky_returns <- c(0.05, -0.02, 0.03, -0.01, 0.07)
safe_returns <- rep(0.01, 5)  # Constant safe asset return
initial_floor <- 0.8
multiplier <- 3
dates <- as.Date(c("2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"))

cppi_result <- cppi_strategy(risky_returns, safe_returns, initial_floor, multiplier, dates)
cppi_result
