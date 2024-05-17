library(tidyverse)
library(lubridate)
library(PerformanceAnalytics)
library(TTR)
library(ggplot2)
library(DT)
library(xts)
library(dplyr)
library(dplyr)
library(zoo) # For as.yearqtr
library(lubridate)
library(tseries)
library(ISOweek)
library(reshape2)
library(data.table)
library(quantmod)


vix_data <- read.csv("vix_data2.csv")

read_and_preprocess_data <- function(file_path) {
  prices <- read.csv(file_path)
  prices$Dates <- as.Date(prices$Dates, format="%m/%d/%Y")
  return(prices)
}


data_cleaning <- function(prices) {
  # Iterate over all columns except for the first (Dates) column
  for (col_name in names(prices)[-1]) {
    # Convert the column to numeric, forcing non-convertible values to NA
    # This will convert '#N/A' and other non-numeric values to NA
    prices[[col_name]] <- as.numeric(as.character(prices[[col_name]]))
    # Check for any NA values and replace them with the previous day's value
    for (i in 2:nrow(prices)) {
      if (is.na(prices[i, col_name])) {
        prices[i, col_name] <- prices[i - 1, col_name]
      }
    }
  }
  # Warning if NAs are found in the first row, as they cannot be replaced by the previous day's value
  if (sum(sapply(prices[1, -1], is.na)) > 0) {
    warning("NAs found in the first row cannot be replaced by previous day's value.")
  }
  return(prices)
}


convert_cumulative_to_daily_returns <- function(prices) {
  # Initialize a new data frame with the Dates column from the given prices data frame
  daily_returns <- data.frame(Dates = prices$Dates)
  # Iterate over all columns in the prices data frame except for the Dates column
  for (col_name in names(prices)[-1]) {  # -1 excludes the first column
    # Convert cumulative returns to price
    price_data <- 100 * (1 + (prices[[col_name]] / 100))
    # Calculate daily returns for the price data
    # Using diff() to calculate the change from the previous day and dividing by the value of the previous day
    daily_return <- c(NA, diff(price_data) / head(price_data, -1))
    # Add the calculated daily returns as a new column to the daily_returns data frame
    # Naming the column with original column name appended with '_Daily_Return'
    daily_return_col_name <- paste(col_name, "Daily_Return", sep = "_")
    daily_returns[[daily_return_col_name]] <- daily_return
    for (col_name in names(daily_returns)[-1]) {  # -1 excludes the first column ('Dates')
      daily_returns[1, col_name] <- 0
    }
  }
  return(daily_returns)
}



file_path <- "SPX as of May 12 20241.csv"
prices <- read_and_preprocess_data(file_path)
prices <- data_cleaning(prices)
most_recent_date <- max(prices$Dates)
cutoff_date <- as.Date(most_recent_date)
# Filter the dataset to include only data after the cutoff date
prices_filtered <- prices %>%
  filter(Dates > cutoff_date)
daily_returns <- convert_cumulative_to_daily_returns(prices)


file_path2 <- "Robbie Benchmark.csv"
prices2 <- read_and_preprocess_data(file_path2)
prices2 <- data_cleaning(prices2)
most_recent_date2 <- max(prices2$Dates)
cutoff_date2 <- as.Date(most_recent_date)
# Filter the dataset to include only data after the cutoff date
prices_filtered2 <- prices2 %>%
  filter(Dates > cutoff_date2)
daily_returns2 <- convert_cumulative_to_daily_returns(prices2)











# Convert date format if not already done
daily_returns$Dates <- as.Date(daily_returns$Dates)

# Convert data from wide to long format
long_daily_returns <- daily_returns %>%
  gather(Stock, return, -Dates) %>%
  mutate(month = month(Dates), year = year(Dates)) %>%
  arrange(Stock, Dates)

# Calculate RSI for each day, then average it monthly
daily_rsi <- long_daily_returns %>%
  group_by(Stock, year, month) %>%
  mutate(previous_return = lag(return, default = first(return)),
         gain = ifelse(return > previous_return, return - previous_return, 0),
         loss = ifelse(return < previous_return, previous_return - return, 0)) %>%
  mutate(average_gain = cummean(gain),  # Initialize with cummean for the first 14 days
         average_loss = cummean(loss)) %>%
  mutate(average_gain = ifelse(row_number() > 14, (lag(average_gain) * 13 + gain) / 14, average_gain),
         average_loss = ifelse(row_number() > 14, (lag(average_loss) * 13 + loss) / 14, average_loss),
         RSI = 100 - (100 / (1 + (average_gain / ifelse(average_loss == 0, 1, average_loss))))) %>%
  select(Stock, Dates, month, year, RSI)

# Aggregate RSI at the monthly level
monthly_rsi <- daily_rsi %>%
  group_by(Stock, year, month) %>%
  summarize(average_monthly_RSI = mean(RSI, na.rm = TRUE)) %>%
  ungroup()

# Calculate monthly returns
monthly_returns <- long_daily_returns %>%
  group_by(Stock, year, month) %>%
  summarize(monthly_return = prod(1 + return) - 1) %>%
  ungroup()

# Merge monthly RSI with monthly returns
monthly_returns_rsi <- monthly_returns %>%
  left_join(monthly_rsi, by = c("Stock", "year", "month"))

# This resulting dataframe 'monthly_returns_rsi' contains both the monthly returns and the average monthly RSI.

# Convert data from wide to long format
daily_returns_long <- daily_returns %>%
  pivot_longer(
    cols = -Dates,  # Keep the Dates column
    names_to = "Stock",  # This will become the stock column
    values_to = "return"  # This will contain return values
  )

selected_stocks_df <- data.frame(Year = integer(), Month = integer(), Type = character(), Stock = character(), stringsAsFactors = FALSE)

for(i in unique(monthly_returns_rsi$month)) {
  for(j in unique(monthly_returns_rsi$year)) {
    # Filter the data for the current month and year, excluding rows where RSI is NA
    current_month_data <- monthly_returns_rsi %>%
      filter(month == i, year == j) %>%
      filter(!is.na(average_monthly_RSI))  # Exclude stocks with NA RSI values
    
    if(nrow(current_month_data) == 0) next
    
    # Rank stocks based on RSI, higher RSI indicates stronger momentum
    ranked_stocks <- current_month_data %>% 
      arrange(desc(average_monthly_RSI))
    n_stocks <- nrow(ranked_stocks)
    
    # Ensure there's enough data
    if(n_stocks < 10) next
    
    # Calculate the number for top 20%
    n_top <- max(1, ceiling(0.20 * n_stocks))
    
    # Select top stocks based on RSI
    top_stocks <- head(ranked_stocks, n_top)
    
    # Append top stocks to the data frame
    top_stocks_df <- data.frame(Year = j, Month = i, Type = "Long", Stock = top_stocks$Stock, stringsAsFactors = FALSE)
    selected_stocks_df <- rbind(selected_stocks_df, top_stocks_df)
  }
}

# This dataframe 'selected_stocks_df' now contains only the top 20% of stocks each month, selected for long positions.




























# Convert data from wide to long format
daily_returns_long <- daily_returns %>%
  pivot_longer(
    cols = -Dates,  # Keep the Dates column
    names_to = "Stock",  # This will become the stock column
    values_to = "return"  # This will contain return values
  )

# Calculate monthly volatility
monthly_volatility <- daily_returns_long %>%
  mutate(month = month(Dates), year = year(Dates)) %>%
  group_by(Stock, year, month) %>%
  summarize(volatility = sd(return, na.rm = TRUE), .groups = 'drop')  # Standard deviation of returns

selected_stocks_df <- data.frame(Year = integer(), Month = integer(), Type = character(), Stock = character(), stringsAsFactors = FALSE)

for(i in unique(monthly_volatility$month)) {
  for(j in unique(monthly_volatility$year)) {
    # Filter the data for the current month and year
    current_month_data <- monthly_volatility %>%
      filter(month == i, year == j)
    
    if(nrow(current_month_data) == 0) next
    
    # Rank stocks based on volatility, lower volatility indicates lower risk
    ranked_stocks <- current_month_data %>% 
      arrange(volatility)
    n_stocks <- nrow(ranked_stocks)
    
    # Ensure there's enough data
    if(n_stocks < 10) next
    
    # Calculate the number for bottom 20% (least volatile)
    n_bottom <- max(1, ceiling(0.20 * n_stocks))
    
    # Select bottom stocks based on volatility
    bottom_stocks <- head(ranked_stocks, n_bottom)
    
    # Append bottom stocks to the data frame
    bottom_stocks_df <- data.frame(Year = j, Month = i, Type = "Low Volatility", Stock = bottom_stocks$Stock, stringsAsFactors = FALSE)
    selected_stocks_df <- rbind(selected_stocks_df, bottom_stocks_df)
  }
}

# This dataframe 'selected_stocks_df' now contains only the least volatile 20% of stocks each month, selected for the MinVar portfolio.





library(dplyr)
library(lubridate)
library(tidyr)
library(TTR)  # Ensure TTR is loaded for RSI calculation

# Assuming daily_returns_long is preprocessed with 'Dates', 'Stock', and 'return'
# Calculate RSI and Volatility within safe checks
monthly_metrics <- daily_returns_long %>%
  group_by(Stock, year = year(Dates), month = month(Dates)) %>%
  summarize(
    monthly_return = prod(1 + return) - 1,
    volatility = sd(return, na.rm = TRUE),
    RSI = if (sum(!is.na(return)) >= 15) mean(RSI(return, n = 14, wilder = FALSE), na.rm = TRUE) else NA_real_,
    .groups = 'drop'
  )




vix_data$Dates <- as.Date(vix_data$Dates, format = "%m/%d/%Y")  # Adjust the format as per your actual data
library(dplyr)
library(lubridate)

vix_data <- vix_data %>%
  mutate(year = year(Dates), month = month(Dates)) %>%
  group_by(year, month) %>%
  summarize(average_VIX = mean(PX_LAST), .groups = 'drop')  # Ensure PX_LAST is correctly named and used

print(head(vix_data))  # Check the first few rows of the processed VIX data

# Read and preprocess VIX data
#vix_data <- read.csv("vix_data2.csv") %>%
  #$mutate(Dates = as.Date(Dates, format="%Y-%m-%d")) %>%
  #mutate(month = month(Dates), year = year(Dates)) %>%
 # group_by(year, month) %>%
  #summarize(average_VIX = mean(PX_LAST), .groups = 'drop')

# Join VIX data with stock metrics
monthly_metrics <- monthly_metrics %>%
  left_join(vix_data, by = c("year", "month"))

# Initialize empty dataframe for selected stocks
selected_stocks_df <- tibble()

# Select stocks based on strategy determined by VIX
monthly_metrics <- monthly_metrics %>%
  group_by(year, month) %>%
  mutate(
    Type = if_else(average_VIX > 20, "Low Volatility", "High Momentum"),
    Rank = case_when(
      Type == "Low Volatility" ~ rank(volatility),
      Type == "High Momentum" ~ rank(-RSI)
    )
  ) %>%
  ungroup()

# Loop through each month and year, applying selection based on type
for (i in unique(monthly_metrics$month)) {
  for (j in unique(monthly_metrics$year)) {
    # Filter the data for the current month and year
    current_month_data <- monthly_metrics %>%
      filter(month == i, year == j)
    
    if (nrow(current_month_data) == 0) next
    
    n_top <- ceiling(0.20 * nrow(current_month_data))
    
    # Select top stocks based on Type
    selected_stocks <- current_month_data %>%
      arrange(Rank) %>%
      slice_head(n = n_top) %>%
      select(Year = year, Month = month, Type, Stock)
    
    # Append selected stocks to the dataframe
    selected_stocks_df <- bind_rows(selected_stocks_df, selected_stocks)
  }
}




selected_stocks_df <- monthly_metrics %>%
  arrange(year, month, Rank) %>%
  group_by(year, month, Type) %>%
  mutate(TopRank = row_number()) %>%
  ungroup() %>%
  group_by(year, month) %>%
  mutate(TopN = ceiling(0.20 * n())) %>%
  filter(TopRank <= TopN) %>%
  select(Year = year, Month = month, Type, Stock) %>%
  mutate(Month = Month + 1,  # Offset for the next month
         Year = if_else(Month > 12, Year + 1, Year),  # Handle year transition
         Month = if_else(Month > 12, Month - 12, Month))

# Adjust 'daily_returns_long' to include year and month for joining
daily_returns_long <- daily_returns_long %>%
  mutate(year = year(Dates), month = month(Dates))

# Join the daily returns with the selected stocks from the previous month
portfolio_daily_returns <- daily_returns_long %>%
  inner_join(selected_stocks_df, by = c("Stock", "year" = "Year", "month" = "Month")) %>%
  group_by(Dates) %>%
  summarize(daily_return = mean(return, na.rm = TRUE)) %>%
  ungroup()

# Initialize portfolio values starting at 100
initial_value <- 100
portfolio_values <- tibble(Dates = unique(daily_returns_long$Dates), PortfolioValue = initial_value)

# Calculate cumulative returns based on daily returns
portfolio_values <- portfolio_values %>%
  left_join(portfolio_daily_returns, by = "Dates") %>%
  arrange(Dates) %>%
  mutate(daily_return = replace_na(daily_return, 0)) %>% # Replace NA returns with 0
  mutate(CumulativeReturn = cumprod(1 + daily_return)) %>% # Compute cumulative factor for returns
  mutate(PortfolioValue = initial_value * CumulativeReturn) # Apply cumulative factor to the initial value


# Print or plot results
print(portfolio_values)


#60/40 STRATEGY
daily_returns2 <- daily_returns2 %>%
  mutate(
    SPX.Cumulative = cumprod(1 + SPX.Index_Daily_Return),
    LBUSTRUU.Cumulative = cumprod(1 + LBUSTRUU.Index_Daily_Return)
  )

# Initial capital and weights
initial_capital <- 100
weights <- c(0.6, 0.4)
# Remove rows where SPX.Index_Daily_Return is NA
daily_returns2 <- daily_returns2 %>%
  filter(!is.na(SPX.Index_Daily_Return))

# Function to rebalance portfolio
rebalance_portfolio <- function(data, initial_capital, weights) {
  n <- nrow(data)
  portfolio_values <- numeric(n)
  portfolio_values[1] <- initial_capital  # Start with initial capital
  
  # Initialize holdings based on initial weights
  spx_holding <- initial_capital * weights[1]
  lbustruu_holding <- initial_capital * weights[2]
  
  for (i in 2:n) {
    # Update holdings based on returns
    spx_holding <- spx_holding * (1 + data$SPX.Index_Daily_Return[i])
    lbustruu_holding <- lbustruu_holding * (1 + data$LBUSTRUU.Index_Daily_Return[i])
    
    # Check if it's the first trading day of a new quarter
    if (month(data$Date[i]) %in% c(1, 4, 7, 10) && (i == 2 || month(data$Date[i-1]) != month(data$Date[i]))) {
      # Rebalance
      total_value <- spx_holding + lbustruu_holding
      spx_holding <- total_value * weights[1]
      lbustruu_holding <- total_value * weights[2]
    }
    
    # Store the total value of the portfolio
    portfolio_values[i] <- spx_holding + lbustruu_holding
  }
  
  return(portfolio_values)
}


# Apply rebalancing function
daily_returns2$Portfolio_Value <- rebalance_portfolio(daily_returns2, initial_capital, weights)

# Plot portfolio value over time
plot(daily_returns2$Date, daily_returns2$Portfolio_Value, type = 'l', col = 'blue', xlab = "Date", ylab = "Portfolio Value", main = "60/40 Portfolio Value Over Time")







# Assume you've loaded S&P 500 daily price data into 'sp500_data'
# Here's an example of converting daily prices to daily returns
# Calculate daily average returns across all stocks
sp500_daily_returns <- daily_returns_long %>%
  group_by(Dates) %>%
  summarize(AverageReturn = mean(return, na.rm = TRUE)) %>%
  ungroup()

# Calculate cumulative returns to mimic the portfolio starting at 100
sp500_daily_returns <- sp500_daily_returns %>%
  mutate(CumulativeReturn = cumprod(1 + AverageReturn)) %>%
  mutate(Sp500PortfolioValue = 100 * CumulativeReturn)  # Starting value is 100

# Ensure portfolio_values contains the right columns (just double-checking)
portfolio_values <- portfolio_values %>%
  select(Dates, PortfolioValue)  # Make sure it has the right columns

# Merge S&P 500 returns with the selected portfolio returns
comparison_data <- left_join(portfolio_values, sp500_daily_returns, by = "Dates")

# Use ggplot2 to visualize the comparison
library(ggplot2)

# Assume 'comparison_data' is already loaded and has columns 'Date', 'PortfolioValue' (momentum/MinVar), 'Sp500PortfolioValue'

# Merge the 60/40 portfolio data into the comparison data
comparison_data <- merge(comparison_data, daily_returns2[, c("Dates", "Portfolio_Value")], by = "Dates", all = TRUE)

# Rename the merged column for clarity
comparison_data <- rename(comparison_data, SixtyFortyPortfolioValue = Portfolio_Value)
# Load necessary library

# Fill NA values by carrying forward the last known value
comparison_data <- comparison_data %>%
  fill(SixtyFortyPortfolioValue, .direction = "downup")

# Plot all three strategies
ggplot(data = comparison_data, aes(x = Dates)) +
  geom_line(aes(y = PortfolioValue, col = "Momentum/MinVar Strategy")) +
  geom_line(aes(y = Sp500PortfolioValue, col = "S&P 500 Portfolio")) +
  geom_line(aes(y = SixtyFortyPortfolioValue, col = "60/40 Portfolio")) +
  labs(title = "Comparison of Portfolio Strategies",
       x = "Date",
       y = "Portfolio Value",
       color = "Strategy") +
  theme_minimal()





# Plotting both the selected portfolio and the S&P 500 portfolio
#ggplot(comparison_data, aes(x = Dates)) +
  #geom_line(aes(y = PortfolioValue, color = "Selected Portfolio")) +
  #geom_line(aes(y = Sp500PortfolioValue, color = "S&P 500 Index")) +
  #labs(title = "Portfolio vs. S&P 500 Performance",
       #x = "Date", y = "Portfolio Value",
       #color = "Portfolio") +
 # scale_color_manual(values = c("Selected Portfolio" = "blue", "S&P 500 Index" = "red")) +
  #theme_minimal()








convert_to_xts <- function(df, date_column = "Dates") {
  date_col <- as.Date(df[[date_column]])
  df <- df[setdiff(names(df), date_column)]
  return(xts(df, order.by = date_col))
}

# Convert your data frames to xts
selected_xts <- convert_to_xts(portfolio_values, date_column = "Dates")
sp500_xts <- convert_to_xts(sp500_daily_returns, date_column = "Dates")






library(dplyr)
library(tidyr)
library(lubridate)
library(PerformanceAnalytics)
library(xts)

# Convert the data into xts objects for financial calculations
selected_xts <- xts(comparison_data$PortfolioValue, order.by = as.Date(comparison_data$Dates))
sp500_xts <- xts(comparison_data$Sp500PortfolioValue, order.by = as.Date(comparison_data$Dates))

# Calculate daily returns for both portfolios
selected_returns <- na.omit(Return.calculate(selected_xts))
sp500_returns <- na.omit(Return.calculate(sp500_xts))



annualized_return_selected <- Return.annualized(selected_returns)
annualized_return_sp500 <- Return.annualized(sp500_returns)

annualized_vol_selected <- sqrt(252) * sd(selected_returns)  # Assuming 252 trading days
annualized_vol_sp500 <- sqrt(252) * sd(sp500_returns)

max_drawdown_selected <- maxDrawdown(selected_returns)
max_drawdown_sp500 <- maxDrawdown(sp500_returns)

sharpe_ratio_selected <- SharpeRatio.annualized(selected_returns, Rf = 0, scale = 252)
sharpe_ratio_sp500 <- SharpeRatio.annualized(sp500_returns, Rf = 0, scale = 252)


fit <- lm(selected_returns ~ sp500_returns)
alpha <- coef(fit)[1]  # Intercept
beta <- coef(fit)[2]   # Slope

yearly_returns_selected <- periodReturn(selected_xts, period = "yearly", type = "log")
yearly_returns_sp500 <- periodReturn(sp500_xts, period = "yearly", type = "log")

# Calculate yearly alpha by subtracting S&P 500 returns from the selected portfolio returns
yearly_alpha <- yearly_returns_selected - yearly_returns_sp500
# Print yearly alpha
print(yearly_alpha)

# Annualize the alpha by multiplying by the number of trading days
annual_alpha <- alpha * 252  # Assuming alpha is daily

cat("Annualized Returns - Selected Portfolio: ", annualized_return_selected, "\n")
cat("Annualized Returns - S&P 500: ", annualized_return_sp500, "\n")
cat("Annualized Volatility - Selected Portfolio: ", annualized_vol_selected, "\n")
cat("Annualized Volatility - S&P 500: ", annualized_vol_sp500, "\n")
cat("Max Drawdown - Selected Portfolio: ", max_drawdown_selected, "\n")
cat("Max Drawdown - S&P 500: ", max_drawdown_sp500, "\n")
cat("Sharpe Ratio - Selected Portfolio: ", sharpe_ratio_selected, "\n")
cat("Sharpe Ratio - S&P 500: ", sharpe_ratio_sp500, "\n")
cat("Alpha: ", alpha, "\n")
cat("Beta: ", beta, "\n")
cat("Annual Alpha: ", annual_alpha, "\n")









#60/40 comparison
selected_xts <- xts(comparison_data$PortfolioValue, order.by = as.Date(comparison_data$Dates))
eb6040_xts <- xts(comparison_data$SixtyFortyPortfolioValue, order.by = as.Date(comparison_data$Dates))

# Calculate daily returns for both portfolios
selected_returns <- na.omit(Return.calculate(selected_xts))
eb6040_returns <- na.omit(Return.calculate(eb6040_xts))



annualized_return_selected <- Return.annualized(selected_returns)
annualized_return_6040 <- Return.annualized(eb6040_returns)

annualized_vol_selected <- sqrt(252) * sd(selected_returns)  # Assuming 252 trading days
annualized_vol_6040 <- sqrt(252) * sd(eb6040_returns)

max_drawdown_selected <- maxDrawdown(selected_returns)
max_drawdown_6040 <- maxDrawdown(eb6040_returns)

sharpe_ratio_selected <- SharpeRatio.annualized(selected_returns, Rf = 0, scale = 252)
sharpe_ratio_6040 <- SharpeRatio.annualized(eb6040_returns, Rf = 0, scale = 252)


fit <- lm(selected_returns ~ eb6040_returns)
alpha <- coef(fit)[1]  # Intercept
beta <- coef(fit)[2]   # Slope

yearly_returns_selected <- periodReturn(selected_xts, period = "yearly", type = "log")
yearly_returns_6040 <- periodReturn(eb6040_xts, period = "yearly", type = "log")

# Calculate yearly alpha by subtracting S&P 500 returns from the selected portfolio returns
yearly_alpha <- yearly_returns_selected - yearly_returns_6040
# Print yearly alpha
print(yearly_alpha)

# Annualize the alpha by multiplying by the number of trading days
annual_alpha <- alpha * 252  # Assuming alpha is daily

cat("Annualized Returns - Selected Portfolio: ", annualized_return_selected, "\n")
cat("Annualized Returns - S&P 500: ", annualized_return_6040, "\n")
cat("Annualized Volatility - Selected Portfolio: ", annualized_vol_selected, "\n")
cat("Annualized Volatility - S&P 500: ", annualized_vol_6040, "\n")
cat("Max Drawdown - Selected Portfolio: ", max_drawdown_selected, "\n")
cat("Max Drawdown - S&P 500: ", max_drawdown_6040, "\n")
cat("Sharpe Ratio - Selected Portfolio: ", sharpe_ratio_selected, "\n")
cat("Sharpe Ratio - S&P 500: ", sharpe_ratio_6040, "\n")
cat("Alpha: ", alpha, "\n")
cat("Beta: ", beta, "\n")
cat("Annual Alpha: ", annual_alpha, "\n")


library(tibble)

# Create a tibble to hold the metrics
financial_metrics_table <- tibble(
  Metric = c("Annualized Returns", "Annualized Volatility", "Max Drawdown", "Sharpe Ratio", "Alpha", "Beta", "Annual Alpha"),
  "Selected Portfolio" = c(annualized_return_selected, 
                           annualized_vol_selected, 
                           max_drawdown_selected, 
                           sharpe_ratio_selected, 
                           alpha, 
                           beta, 
                           annual_alpha),
  "S&P 500" = c(annualized_return_sp500, 
                annualized_vol_sp500, 
                max_drawdown_sp500, 
                sharpe_ratio_sp500,
                NA,  # Alpha and Beta are typically not applicable directly to S&P 500 in this context
                NA,
                NA)
)

# Print the table
print(financial_metrics_table)

# Optionally, you can use the kable function from the knitr package for a nicer formatted table in markdown or HTML outputs
knitr::kable(financial_metrics_table, caption = "Financial Metrics Comparison")




library(dplyr)
library(ggplot2)
library(lubridate)

# Ensure selected_stocks_df has the 'Dates' column; if not, create it:
selected_stocks_df$Dates <- as.Date(paste(selected_stocks_df$Year, selected_stocks_df$Month, "1", sep = "-"))

# Aggregate to get only one row per month (assuming the type doesn't change within a month)
strategy_data <- selected_stocks_df %>%
  group_by(Dates, Year, Month) %>%
  summarize(Type = first(Type), .groups = "drop")

# If your data isn't ordered (important for plotting):
strategy_data <- strategy_data %>%
  arrange(Dates)

ggplot(strategy_data, aes(x = Dates, fill = Type)) +
  geom_bar(stat = "count", width = 20, show.legend = TRUE) +  # Adjust 'width' as necessary
  scale_fill_manual(values = c("Low Volatility" = "blue", "High Momentum" = "red")) +
  labs(title = "Monthly Portfolio Strategy Allocation",
       x = "Date",
       y = "Count of Months",
       fill = "Strategy") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +  # Set breaks every 1 year
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Improve date labels readability

ggplot(strategy_data, aes(x = Dates, y = Type, group = 1)) +
  geom_line(aes(color = Type), size = 1) +
  scale_color_manual(values = c("Low Volatility" = "blue", "High Momentum" = "red")) +
  labs(title = "Portfolio Strategy Over Time",
       x = "Date",
       y = "Strategy Type",
       color = "Strategy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Improve date labels readability



ggplot(strategy_data, aes(x = Dates, y = Type, group = 1)) +
  geom_step(aes(color = Type), size = 1.2) +
  scale_color_manual(values = c("Low Volatility" = "blue", "High Momentum" = "red")) +
  labs(title = "Portfolio Strategy Over Time",
       x = "Date",
       y = "Strategy Type",
       color = "Strategy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Improve date labels readability


yearly_alpha_df <- data.frame(Date = index(yearly_alpha), Alpha = coredata(yearly_alpha))

# Making sure the column names are correct
colnames(yearly_alpha_df) <- c("Date", "Alpha")

# Plotting the yearly alpha
ggplot(yearly_alpha_df, aes(x = Date, y = Alpha, fill = Alpha > 0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_manual(values = c(`FALSE` = "red", `TRUE` = "green")) +  # Specifying colors for TRUE and FALSE directly
  labs(title = "Yearly Alpha Compared to SP500", x = "Year", y = "Alpha") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_minimal()







































































# Assuming monthly_metrics and daily_returns_long are already loaded
# Initialize the portfolio value
initial_value <- 100

# Initialize a data frame to store daily portfolio values
portfolio_values <- data.frame(Dates = unique(daily_returns_long$Dates), PortfolioValue = NA)
portfolio_values$PortfolioValue[1] <- initial_value



library(dplyr)
library(tidyr)
library(lubridate)

# Calculate the monthly selection of stocks based on the provided criteria
selected_stocks_df <- data.frame(Year = integer(), Month = integer(), Type = character(), Stock = character(), stringsAsFactors = FALSE)

monthly_metrics <- monthly_metrics %>%
  arrange(year, month, Rank) %>%
  group_by(year, month, Type) %>%
  mutate(TopRank = row_number()) %>%
  ungroup()

# Calculate the top 20% for each type, each month
library(dplyr)
library(tidyr)
library(lubridate)

# Assume monthly_metrics is already prepared with 'year', 'month', 'Stock', 'Type', and 'Rank'
# First, compute the top stocks for each type for each month
selected_stocks_df <- monthly_metrics %>%
  arrange(year, month, Rank) %>%
  group_by(year, month, Type) %>%
  mutate(TopRank = row_number()) %>%
  ungroup() %>%
  group_by(year, month) %>%
  mutate(TopN = ceiling(0.20 * n())) %>%
  filter(TopRank <= TopN) %>%
  select(Year = year, Month = month, Type, Stock)

# Shift the stock selection to apply for the next month
selected_stocks_df <- selected_stocks_df %>%
  mutate(Month = Month + 1) %>%
  # Handle year transition
  mutate(Year = if_else(Month > 12, Year + 1, Year),
         Month = if_else(Month > 12, Month - 12, Month))

# Assuming daily_returns_long is already loaded
daily_returns_long <- daily_returns_long %>%
  mutate(year = year(Dates), month = month(Dates))

selected_stocks_df <- monthly_metrics %>%
  select(Year = year, Month = month, Type, Stock)




# Calculate daily returns for the portfolio based on the previous month's selected stocks
portfolio_daily_returns <- daily_returns_long %>%
  inner_join(selected_stocks_df, by = c("Stock", "year" = "Year", "month" = "Month")) %>%
  group_by(Dates) %>%
  summarize(daily_return = mean(return, na.rm = TRUE)) %>%
  ungroup()


# Initialize portfolio value
initial_value <- 100
portfolio_values <- data.frame(Dates = unique(daily_returns_long$Dates), PortfolioValue = NA)
portfolio_values$PortfolioValue[1] <- initial_value

# Compute cumulative returns based on daily returns
portfolio_values <- left_join(portfolio_values, portfolio_daily_returns, by = "Dates")
library(dplyr)
library(tidyr)  # For fill()

# Assuming 'portfolio_values' already includes 'Dates' and 'daily_return' from previous steps
portfolio_values <- portfolio_values %>%
  arrange(Dates) %>%
  mutate(daily_return = replace_na(daily_return, 0))  # Replace NA returns with 0 for no change in value

# Calculate the cumulative product of daily returns adjusted by the initial value
portfolio_values$PortfolioValue <- initial_value * cumprod(1 + portfolio_values$daily_return)

# Fill forward any remaining NA values in PortfolioValue if they exist
portfolio_values$PortfolioValue <- tidyr::fill(portfolio_values$PortfolioValue, .direction = "down")



library(ggplot2)

ggplot(portfolio_values, aes(x = Dates, y = PortfolioValue)) +
  geom_line() +
  labs(title = "Cumulative Portfolio Performance", y = "Portfolio Value", x = "Date") +
  theme_minimal()

# Join the daily returns to selected stocks to calculate the portfolio return
daily_returns_long <- daily_returns_long %>%
  mutate(year = year(Dates), month = month(Dates))

portfolio_daily_returns <- daily_returns_long %>%
  inner_join(selected_stocks_df, by = c("Stock", "year" = "Year", "month" = "Month")) %>%
  group_by(Dates) %>%
  summarize(daily_return = mean(return, na.rm = TRUE)) %>%
  ungroup()


# Compute cumulative returns based on daily returns
portfolio_daily_returns <- portfolio_daily_returns %>%
  arrange(Dates) %>%
  mutate(PortfolioValue = initial_value * cumprod(1 + daily_return))

# Merge with the initialized portfolio values data frame
portfolio_values <- left_join(portfolio_values, portfolio_daily_returns, by = "Dates")

# Fill NA values with the last known portfolio value (carry forward last value)
portfolio_values$PortfolioValue <- zoo::na.locf(portfolio_values$PortfolioValue)


library(ggplot2)

# Plot the cumulative portfolio returns over time
ggplot(portfolio_values, aes(x = Dates, y = PortfolioValue)) +
  geom_line() +
  labs(title = "Cumulative Portfolio Performance", y = "Portfolio Value", x = "Date") +
  theme_minimal()






















vix_data$Dates <- as.Date(vix_data$Dates, format="%m/%d/%Y")

# Ensure the Dates conversion has been applied correctly
vix_data <- vix_data %>%
  mutate(Dates = as.Date(Dates, format = "%Y-%m-%d"))
daily_returns_long <- daily_returns %>%
  pivot_longer(cols = -Dates, names_to = "Stock", values_to = "return") %>%
  mutate(
    Dates = as.Date(Dates),
    month = month(Dates), 
    year = year(Dates)
  ) %>%
  arrange(Stock, Dates)


vix_data <- read.csv("vix_data.csv") %>%
  mutate(Dates = as.Date(Dates, format = "%m/%d/%Y")) %>%
  mutate(month = month(Dates), year = year(Dates)) %>%
  group_by(year, month) %>%
  summarize(average_VIX = mean(PX_LAST, na.rm = TRUE), .groups = 'drop')


monthly_metrics <- daily_returns_long %>%
  group_by(Stock, year, month) %>%
  summarize(
    monthly_return = prod(1 + return) - 1,
    volatility = sd(return, na.rm = TRUE),
    RSI = mean(RSI(return, n = 14, wilder = FALSE), na.rm = TRUE), # Calculate RSI, need the TTR package or custom function
    .groups = 'drop'
  )



vix_data <- read.csv("vix_data.csv") %>%
  mutate(Dates = as.Date(Dates, format = "%m/%d/%Y")) %>%
  mutate(month = month(Dates), year = year(Dates)) %>%
  group_by(year, month) %>%
  summarize(average_VIX = mean(PX_LAST, na.rm = TRUE), .groups = 'drop')

# Assuming daily_returns_long is already loaded and prepared
daily_returns_long <- daily_returns %>%
  pivot_longer(cols = -Dates, names_to = "Stock", values_to = "return") %>%
  mutate(Dates = as.Date(Dates),
         month = month(Dates),
         year = year(Dates)) %>%
  arrange(Dates)
daily_returns_long <- daily_returns_long %>%
  left_join(vix_data, by = c("year", "month"))








selected_stocks_df <- daily_returns_long %>%
  group_by(year, month, Stock) %>%
  summarize(
    monthly_return = mean(return, na.rm = TRUE), 
    volatility = sd(return, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    strategy = if_else(average_VIX > 25, "MinVar", "Momentum"),
    selected = case_when(
      strategy == "Momentum" & monthly_return >= quantile(monthly_return, 0.8, na.rm = TRUE) ~ TRUE,
      strategy == "MinVar" & volatility <= quantile(volatility, 0.2, na.rm = TRUE) ~ TRUE,
      TRUE ~ FALSE
    )
  ) %>%
  filter(selected)

# Calculate daily returns for the selected portfolio
portfolio_daily_returns <- daily_returns_long %>%
  filter(Stock %in% selected_stocks_df$Stock) %>%
  group_by(Dates) %>%
  summarize(daily_return = mean(return, na.rm = TRUE), .groups = 'drop')

# Calculate cumulative returns starting from 100
portfolio_daily_returns <- portfolio_daily_returns %>%
  mutate(cumulative_return = 100 * cumprod(1 + daily_return))

# Plot the cumulative returns
ggplot(portfolio_daily_returns, aes(x = Dates, y = cumulative_return)) +
  geom_line() +
  labs(title = "Cumulative Portfolio Returns", x = "Date", y = "Cumulative Return")

# Print the daily returns dataframe
print(portfolio_daily_returns)

