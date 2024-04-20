#--------------- Load Required Packages ---------------
library(quantmod)
library(lubridate)
library(ggplot2)
library(plotly)

#--------------- Load Required Function ---------------
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
source("CPPI.R")

#--------------- Functions ---------------
get_data <- function(ticker, start_date, end_date) {
  getSymbols(ticker, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
}

calculate_returns <- function(prices) {
  returns <- diff(log(Ad(prices)))
  return(c(NA, returns))  # Proper NA handling
}

#--------------- Fetch and Prepare Data ---------------
start_date <- Sys.Date() - years(5)
end_date <- Sys.Date()

# Fetch data
spy_data <- get_data("SPY", start_date, end_date)
agg_data <- get_data("AGG", start_date, end_date)

# Calculate returns
spy_returns <- calculate_returns(spy_data)[-c(1:2)]
agg_returns <- calculate_returns(agg_data)[-c(1:2)]

# Dates for the CPPI function
dates <- index(spy_data)[-1]

#--------------- Apply CPPI Strategy ---------------
floor <- 0.8  # Initial portfolio value
multiplier <- 4  # Level of exposure to the risky asset

# Apply the CPPI strategy
cppi_result <- cppi_strategy(as.numeric(spy_returns), as.numeric(agg_returns), floor, multiplier, dates)

#--------------- Visualize Cumulative Returns ---------------
cum_spy_returns <- cumprod(1 + spy_returns[-1]) - 1
cum_agg_returns <- cumprod(1 + agg_returns[-1]) - 1
cum_cppi_returns <- cumprod(1 + cppi_result$portfolio_returns[-1]) - 1

cum_returns_df <- data.frame(Date = dates[-1], SPY = cum_spy_returns, AGG = cum_agg_returns, CPPI = cum_cppi_returns)

# Plot using ggplot2
ggplot(cum_returns_df, aes(x = Date)) +
  geom_line(aes(y = SPY, colour = "SPY"), size = 1) +
  geom_line(aes(y = AGG, colour = "AGG"), size = 1) +
  geom_line(aes(y = CPPI, colour = "CPPI"), size = 1) +
  labs(title = "Cumulative Returns Comparison", y = "Cumulative Returns", x = "Date") +
  scale_colour_manual("", values = c("SPY" = "black", "AGG" = "grey", "CPPI" = "blue")) +
  theme_minimal()

#--------------- Visualize Allocations ---------------
#--------------- Prepare Data ---------------
allocation_df <- data.frame(Date = dates, Risky = cppi_result$allocation_risky, Safe = cppi_result$allocation_safe)

# Calculate the percentage allocations
allocation_df$Total = allocation_df$Risky + allocation_df$Safe
allocation_df$Pct_Risky = allocation_df$Risky / allocation_df$Total * 1000
allocation_df$Pct_Safe = allocation_df$Safe / allocation_df$Total * 1000

# Reshape data for stacking in ggplot2
allocation_long <- reshape2::melt(allocation_df, id.vars = "Date", measure.vars = c("Pct_Risky", "Pct_Safe"))

ggplot(allocation_long, aes(x = Date, y = value, fill = variable)) +
  geom_area(position = 'stack') +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "CPPI Allocations Over Time",
    x = "Date",
    y = "Allocation",
    fill = "Asset Class"
  ) +
  scale_fill_manual(values = c("Pct_Risky" = "blue", "Pct_Safe" = "grey")) +
  theme_minimal()

