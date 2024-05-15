# file: CPPI.R
# version: 1.03

#--------------- CPPI Strategy Function ---------------
cppi_strategy <- function(risky_returns, safe_returns, dates, floor = 0.8, multiplier = 3, floor_time_int = 250) {
  initial_value <- 100
  n <- length(risky_returns)
  portfolio_value <- rep(initial_value, n)
  cushion <- rep(0, n)
  allocation_risky <- rep(0, n)
  allocation_safe <- rep(0, n)
  
  # Initial floor value setup
  floor_values <- rep(initial_value * floor, n)  # Create a vector to store floor values over time
  last_update_index <- 1  # Tracks last index where floor was updated
  
  for (t in 1:(n-1)) {
    # Check if the floor should be updated based on the specified time interval
    if (t >= last_update_index + floor_time_int || t == 1) {
      floor_values[t] <- max(portfolio_value[t] * floor, floor_values[t - 1])
      last_update_index <- t
    } else {
      floor_values[t] <- floor_values[t - 1]  # Carry forward the last updated floor value
    }
    
    cushion[t] <- max(portfolio_value[t] - floor_values[t], 0)
    allocation_risky[t] <- min(cushion[t] * multiplier, portfolio_value[t])
    allocation_safe[t] <- portfolio_value[t] - allocation_risky[t]
    portfolio_value[t + 1] <- allocation_risky[t] * (1 + risky_returns[t]) + allocation_safe[t] * (1 + safe_returns[t])
  }
  
  # For the last element in floor_values, carry forward from the last update if not updated at end
  if (length(floor_values) > 1 && (last_update_index + floor_time_int) > n) {
    floor_values[n] <- floor_values[n - 1]
  }
  
  portfolio_returns <- c(NA, diff(portfolio_value) / portfolio_value[-n])
  return(list(portfolio_returns = portfolio_returns,
              allocation_risky = allocation_risky,
              allocation_safe = allocation_safe,
              floor_values = floor_values))  # Return the complete vector of floor values
}

# Example usage with hypothetical returns and parameters
risky_returns <- c(0.05, -0.02, 0.03, -0.01, 0.07)
safe_returns <- rep(0.01, 5)  # Constant safe asset return
floor  <- 0.8
multiplier <- 3
dates <- as.Date(c("2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"))

cppi_result <- cppi_strategy(
  risky_returns = risky_returns, 
  safe_returns = safe_returns, 
  dates = dates, 
  floor = floor, 
  multiplier = multiplier
)
cppi_result

