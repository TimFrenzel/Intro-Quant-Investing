#Project: InvestmentPortfolio
#Version: 1.19

#--------------- Library Management ---------------
packages_needed <- c("renv", "shiny", "shinyjs", "shinydashboard", "quantmod", "xts", "zoo","ROI","DT","dplyr","PerformanceAnalytics", "TTR", "PortfolioAnalytics","plotly", "lubridate", "ggplot2", "reshape2", "shinythemes","gridExtra")
new_packages <- packages_needed[!packages_needed %in% installed.packages()[, "Package"]]
if (length(new_packages)) install.packages(new_packages)
lapply(packages_needed, library, character.only = TRUE)

#--------------- Function Management ---------------
# Source the CPPI Strategy function
source("/Users/Ziggy/Documents/FIN 496/RL_project/MomentumBot/Intro-Quant-Investing/AWS Deployed/Investment Strategies/CPPI/code/Momentum.R")

#--------------- Data Management ---------------
# Define a list of ETF tickers
etf_tickers <- c("SPY", "AGG", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD")


#--------------- Utility Functions ---------------
isNewQuarter <- function(date, previousDate) {
  month(date) %in% c(1, 4, 7, 10) && month(date) != month(previousDate)
}

getNormalizedWeights <- function(symbol_weights) {
  # Convert input to numeric vector if it's not already
  if(is.list(symbol_weights)) {
    symbol_weights <- unlist(symbol_weights)
  }
  symbol_weights <- as.numeric(symbol_weights)
  
  total_weight <- sum(symbol_weights, na.rm = TRUE)
  if (total_weight > 0) {
    return(symbol_weights / total_weight)
  } else {
    return(rep(1 / length(symbol_weights), length(symbol_weights)))
  }
}

simulatePortfolio <- function(initial_weights, combined_returns) {
  if(is.null(initial_weights) || length(initial_weights) == 0 || sum(is.na(initial_weights)) > 0 || sum(initial_weights) == 0) {
    return(NULL)  # Return NULL or another safe value indicating an error or incomplete state
  }
  
  num_assets <- length(initial_weights)
  portfolio_values <- rep(NA, nrow(combined_returns))
  weights <- matrix(NA, nrow = nrow(combined_returns), ncol = num_assets, dimnames = list(NULL, names(initial_weights)))
  
  # Ensure initial_weights is not empty or null
  if(is.null(initial_weights) || length(initial_weights) == 0) {
    stop("initial_weights is empty or not defined.")
  }
  
  portfolio_values[1] <- 100
  weights[1, ] <- initial_weights
  
  if (nrow(combined_returns) > 1) { # Check if there are enough rows to perform the operation
    for (i in 2:nrow(combined_returns)) {
      asset_values <- portfolio_values[i - 1] * weights[i - 1, ] * (1 + combined_returns[i, ])
      portfolio_values[i] <- sum(asset_values)
      weights[i, ] <- asset_values / portfolio_values[i]
      
      if (isNewQuarter(index(combined_returns)[i], index(combined_returns)[i - 1])) {
        weights[i, ] <- initial_weights
      }
    }
  } else {
    warning("combined_returns does not have enough data.")
  }
  
  return(list(values_xts = xts(portfolio_values, order.by=index(combined_returns)), weights_xts = xts(weights, order.by=index(combined_returns))))
}

#--------------- UI Definition ---------------
# Define the JavaScript code
jsCode <- "shinyjs.runAnalysis = function() { $('#runAnalysis').click(); }"

ui <- dashboardPage(
  dashboardHeader(title = "Investment Portfolio Analysis"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Analysis", tabName = "analysis", icon = icon("home")),
      menuItem("Allocation", tabName = "allocation", icon = icon("area-chart")),
      menuItem("Momentum Strategy", tabName = "momentumStrategy", icon = icon("dashboard"))
    ),
    actionButton("runAnalysis", "Run Analysis", class = "btn-primary"),
    dateInput("startDate", "Start Date", value = "2023-12-31"),
    dateInput("endDate", "End Date", value = Sys.Date()),
    selectInput("symbols", "Symbols", choices = etf_tickers, multiple = TRUE, selected = c("SPY", "AGG")),
    uiOutput("benchmarkWeightsUI"),
    uiOutput("fundWeightsUI")
  ),
  dashboardBody(
    useShinyjs(),
    extendShinyjs(text = jsCode, functions = c("runAnalysis")),
    tags$head(tags$script(src = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML")),
    
    tabItems(
      # Tab for Analysis
      tabItem(tabName = "analysis",
              conditionalPanel(
                condition = "output.loading === true",
                tags$div(class = "loading-message", "Loading data... Please wait.")
              ),
              fluidRow(
                box(plotOutput("performanceComparisonPlot"), title = "Performance Comparison", width = 12)
              ),
              fluidRow(
                box(DT::dataTableOutput("performanceMetricsTable"), title = "Performance Metrics", width = 12)
              )
      ),
      
      # Tab for Allocation
      tabItem(tabName = "allocation",
              fluidRow(
                box(plotOutput("benchmarkAllocationPlot"), title = "Benchmark Allocation", width = 6),
                box(plotOutput("fundAllocationPlot"), title = "Fund Allocation", width = 6)
              )
      ),
      
      # Tab for Momentum Strategy
      tabItem(tabName = "momentumStrategy",
              div(style = "margin-left: 20px;",  # Margin for layout spacing
                  fluidRow(
                    selectInput("Asset1", "Select First Asset", choices = etf_tickers, selected = c("SPY")),
                    selectInput("Asset2", "Select Second Asset", choices = etf_tickers, selected = c("AGG"))
                  ),
                  fluidRow(
                    selectInput("lookbackPeriod", "Set lookback period", choices = c(6, 12, 24, 36)),
                    sliderInput("allocationChange", "Set Multiplier", min = 0.5, max = 15, value = 3, step = 0.5)
                  ),
                  fluidRow(
                    dateRangeInput("dateRange", "Select Date Range", start = Sys.Date() - 365, end = Sys.Date())
                  ),
                  uiOutput("errorDisplay"),
                  plotOutput("momentumResults"),
                  plotOutput("allocationPlot"),
                  htmlOutput("momentumMetrics")
              )
      )
    ),
    
    # Additional UI elements such as author label and version number
    absolutePanel(
      bottom = 10, right = 10, 
      style = "background: transparent; color: #555; font-size: 9px;",
      HTML("Author: Ziggy Molteni <br> Version: 1.19")
    ),
    
    # Theme setting for the dashboard
    theme = shinytheme("sandstone")
  )
)

#--------------- Server Logic ---------------
server <- function(input, output, session) {
  # Initialize reactive value for loading state
  loading_state <- reactiveVal(TRUE)
  errorText <- reactiveVal("")
  
  # Properly define outputs for error message display
  output$errorDisplay <- renderUI({
    if (!is.null(errorText()) && errorText() != "") {
      div(style = "color: red;", strong("Error: "), errorText())
    }
  })
  
  # Output for conditionalPanel to listen to for showing/hiding loading message
  output$loading <- reactive({ loading_state() })
  outputOptions(output, "loading", suspendWhenHidden = FALSE)
  
  # Set default values and trigger data loading immediately
  observe({
    updateSelectInput(session, "symbols", selected = c("SPY", "AGG"))
    
    # Ensure default weights are set for both benchmark and fund weights
    lapply(c("SPY", "AGG"), function(sym) {
      updateNumericInput(session, paste0("benchmarkWeight", sym), value = ifelse(sym == "SPY", 0.6, 0.4))
      updateNumericInput(session, paste0("fundWeight", sym), value = ifelse(sym == "SPY", 1, 0))
    })
    
    # Set loading state to TRUE right before fetching data
    loading_state(TRUE)
    
    # Programmatic click to trigger analysis
    shinyjs::runjs('$("#runAnalysis").click();')
  })
  
  # UI for setting benchmark weights
  output$benchmarkWeightsUI <- renderUI({
    req(input$symbols)
    lapply(input$symbols, function(sym) {
      numericInput(inputId = paste0("benchmarkWeight", sym),
                   label = paste("Benchmark Weight for", sym),
                   value = ifelse(sym == "SPY", 0.6, ifelse(sym == "AGG", 0.4, 1 / length(input$symbols))),
                   min = 0, max = 1, step = 0.01)
    })
  })
  
  # UI for setting fund weights
  output$fundWeightsUI <- renderUI({
    req(input$symbols)
    lapply(input$symbols, function(sym) {
      numericInput(inputId = paste0("fundWeight", sym),
                   label = paste("Fund Weight for", sym),
                   value = ifelse(sym == "SPY", 1, 0),  # Default to 100% SPY, 0% everything else
                   min = 0, max = 1, step = 0.01)
    })
  })
  
  # Reactive expression to fetch symbol data based on user inputs
  symbolData <- eventReactive(input$runAnalysis, {
    req(input$symbols, input$startDate, input$endDate)
    
    # Attempt to fetch data, with error handling
    data <- tryCatch({
      lapply(input$symbols, function(sym) {
        getSymbols(sym, src = "yahoo", from = input$startDate, to = input$endDate, auto.assign = FALSE)
      })
    }, error = function(e) {
      # On error, log and return NULL to prevent downstream errors
      print(paste("Error fetching symbol data:", e$message))
      NULL
    })
    
    # Data fetching complete, set loading state to FALSE
    loading_state(FALSE)
    
    # Return fetched data
    data
  }, ignoreNULL = FALSE)
  
  combinedReturns <- eventReactive(input$runAnalysis, {
    req(symbolData())
    daily_returns <- lapply(symbolData(), function(data) dailyReturn(Cl(data)))
    do.call(merge.xts, daily_returns)
  }, ignoreNULL = FALSE)
  
  portfolioSimulation <- eventReactive(input$runAnalysis, {
    # Use req to ensure that all required inputs are available before proceeding
    req(input$symbols, input$startDate, input$endDate)
    req(sapply(input$symbols, function(sym) input[[paste0("benchmarkWeight", sym)]]))
    req(sapply(input$symbols, function(sym) input[[paste0("fundWeight", sym)]]))
    
    benchmark_weights <- sapply(input$symbols, function(sym) as.numeric(input[[paste0("benchmarkWeight", sym)]]))
    fund_weights <- sapply(input$symbols, function(sym) as.numeric(input[[paste0("fundWeight", sym)]]))
    
    benchmark_weights <- getNormalizedWeights(benchmark_weights)
    fund_weights <- getNormalizedWeights(fund_weights)
    
    benchmark_portfolio <- simulatePortfolio(benchmark_weights, combinedReturns())
    fund_portfolio <- simulatePortfolio(fund_weights, combinedReturns())
    
    list(benchmark = benchmark_portfolio, fund = fund_portfolio)
  }, ignoreNULL = FALSE)
  
  output$benchmarkAllocationPlot <- renderPlot({
    req(portfolioSimulation())
    weights_xts <- portfolioSimulation()$benchmark$weights_xts
    weights_df <- as.data.frame(weights_xts)
    weights_df$Date <- index(weights_xts)
    # Ensure symbols are properly named in the melted data
    colnames(weights_df)[-length(colnames(weights_df))] <- input$symbols
    weights_long <- reshape2::melt(weights_df, id.vars = "Date", variable.name = "Symbol", value.name = "Weight")
    
    ggplot(weights_long, aes(x = Date, y = Weight, fill = Symbol)) +
      geom_area(position = 'stack') +
      scale_fill_brewer(palette = "Pastel1") +
      theme_minimal() +
      labs(x = "", y = "") +
      scale_y_continuous(labels = scales::percent_format()) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.title = element_blank())
  })
  
  output$fundAllocationPlot <- renderPlot({
    req(portfolioSimulation())
    weights_xts <- portfolioSimulation()$fund$weights_xts
    weights_df <- as.data.frame(weights_xts)
    weights_df$Date <- index(weights_xts)
    # Ensure symbols are properly named in the melted data
    colnames(weights_df)[-length(colnames(weights_df))] <- input$symbols
    weights_long <- reshape2::melt(weights_df, id.vars = "Date", variable.name = "Symbol", value.name = "Weight")
    
    ggplot(weights_long, aes(x = Date, y = Weight, fill = Symbol)) +
      geom_area(position = 'stack') +
      scale_fill_brewer(palette = "Set3") +
      theme_minimal() +
      labs(x = "", y = "") +
      scale_y_continuous(labels = scales::percent_format()) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.title = element_blank())
  })
  
  output$performanceComparisonPlot <- renderPlot({
    req(portfolioSimulation())
    benchmark_values <- as.numeric(portfolioSimulation()$benchmark$values_xts)
    fund_values <- as.numeric(portfolioSimulation()$fund$values_xts)
    comparison_df <- data.frame(Date = index(portfolioSimulation()$benchmark$values_xts), Benchmark = benchmark_values, Fund = fund_values)
    comparison_long <- reshape2::melt(comparison_df, id.vars = 'Date')
    
    ggplot(comparison_long, aes(x = Date, y = value, color = variable)) +
      geom_line() +
      theme_minimal() +
      labs(x = "Date", y = "Portfolio Value") +
      scale_color_manual(values = c("Benchmark" = "#2C3E50", "Fund" = "#E74C3C")) +
      theme(legend.title = element_blank(), legend.position = "top")
  })
  
  #--------------- Strategies ---------------
  # CPPI Strategy
  output$momentumResults <- renderPlot({
    req(input$Asset1, input$Asset2, input$lookbackPeriod, input$allocationChange, input$dateRange)
    
    tryCatch({
      # Fetching data for the first two stocks
      asset1_data <- getSymbols(input$Asset1, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      asset2_data <- getSymbols(input$Asset2, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      
      lookbackPeriod <- as.numeric(input$lookbackPeriod)
      allocationChange <- as.numeric(input$allocationChange)
      
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
      
      
      momentumStrategy <- momentum_strategy(asset1_data, asset2_data, lookbackPeriod, allocationChange)
      
      # Calculate daily returns
      asset1_returns <- dailyReturn(Cl(asset1_data), type = 'log')
      asset2_returns <- dailyReturn(Cl(asset2_data), type = 'log')
      
      # Cumulative returns calculation
      asset1_cumreturns <- exp(cumsum(asset1_returns)) - 1  # Convert log returns to cumulative returns
      asset2_cumreturns <- exp(cumsum(asset2_returns)) - 1
      
      # Prepare the data frames for plotting
      cum_returns_df <- data.frame(
        Date = index(momentumStrategy),
        StrategyCumulativeReturns = as.numeric(momentumStrategy$Cumulative_Returns)
      )
      
      # Plotting the data using ggplot2
      gg <- ggplot(cum_returns_df, aes(x = Date)) +
        geom_line(aes(y = StrategyCumulativeReturns, color = "Strategy Cumulative Returns"), size = 1.5, linetype = "dashed") +
        scale_color_manual(values = c("Strategy Cumulative Returns" = "blue")) +
        scale_y_continuous(name = "Cumulative Returns", labels = scales::percent_format()) +
        theme_minimal() +
        labs(title = "Cumulative Returns Comparison", y = "Cumulative Returns", x = "Date")
      
      gg
    }, error = function(e) {
      print(e$message)
      NULL
    })
  })
  
  output$allocationPlot <- renderPlot({
    req(input$Asset1, input$Asset2, input$lookbackPeriod, input$allocationChange, input$dateRange)
    
    tryCatch({
      asset1_data <- getSymbols(input$Asset1, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      asset2_data <- getSymbols(input$Asset2, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      
      lookbackPeriod <- as.numeric(input$lookbackPeriod)
      allocationChange <- as.numeric(input$allocationChange)
      
      results <- momentum_strategy(asset1_data, asset2_data, lookbackPeriod, allocationChange)
      
      # Prepare data for plotting
      allocation_df <- data.frame(
        Date = results$Date,
        Stock1_Allocation = results$Stock1_Allocation,
        Stock2_Allocation = results$Stock2_Allocation
      )
      allocation_long <- reshape2::melt(allocation_df, id.vars = "Date", variable.name = "Stock", value.name = "Allocation")
      
      # Plotting using ggplot2
      ggplot(allocation_long, aes(x = Date, y = Allocation, fill = Stock)) +
        geom_bar(stat = "identity", position = "stack") +  # Use geom_bar with stat="identity" for pre-summarized data
        scale_fill_brewer(palette = "Pastel1") +
        theme_minimal() +
        labs(title = "Allocation Over Time", x = "Date", y = "Allocation (%)") +
        scale_y_continuous(labels = scales::percent_format()) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.title = element_blank())
    }, error = function(e) {
      print(e$message)
      NULL
    })
  })
  
  output$momentumMetrics <- renderText({
    req(input$Asset1, input$Asset2, input$lookbackPeriod, input$allocationChange, input$dateRange)
    
    tryCatch({
      # Fetching data for the two stocks and running the momentum strategy
      asset1_data <- getSymbols(input$Asset1, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      asset2_data <- getSymbols(input$Asset2, src = "yahoo", from = input$dateRange[1], to = input$dateRange[2], auto.assign = FALSE)
      lookbackPeriod <- as.numeric(input$lookbackPeriod)
      allocationChange <- as.numeric(input$allocationChange)
      momentumStrategyResults <- momentum_strategy(asset1_data, asset2_data, lookbackPeriod, allocationChange)
      
      approx_daily_returns <- (1 + momentumStrategyResults$Portfolio_Returns)^(1/30) - 1
      
      # Calculate Total Return for the portfolio
      portfolio_total_return <- Return.cumulative(approx_daily_returns, geometric = TRUE) * 100
      
      # Calculate Volatility for the portfolio
      portfolio_volatility <- sd(approx_daily_returns) * sqrt(252) * 100
      
      # Calculate Sharpe Ratio for the portfolio, assuming a risk-free rate of 0%
      portfolio_sharpe_ratio <- (mean(approx_daily_returns) / sd(approx_daily_returns)) * sqrt(252)
      
      # Create output text with Total Return, Volatility, and Sharpe Ratio
      output_text <- sprintf("<span style='font-size:16px;'>Portfolio Metrics<br/>Total Return: %.2f%%<br/>Volatility: %.2f%%<br/>Sharpe Ratio: %.2f<br/></span>",
                             portfolio_total_return, 
                             portfolio_volatility, 
                             portfolio_sharpe_ratio)
      
      return(HTML(output_text))  # Ensure that HTML is correctly rendered
    }, error = function(e) {
      print(e$message)
      "Error in calculating total returns"
    })
  })
  
  
  
  
  
  
  #--------------- Performance Metrics Table Rendering ---------------
  output$performanceMetricsTable <- DT::renderDataTable({
    sim_results <- req(portfolioSimulation())
    req(!is.null(sim_results$benchmark$values_xts) && !is.null(sim_results$fund$values_xts))
    
    benchmark_returns <- dailyReturn(portfolioSimulation()$benchmark$values_xts)
    fund_returns <- dailyReturn(portfolioSimulation()$fund$values_xts)
    risk_free_rate <- 0  # Assuming risk-free rate is 0 for Beta calculation, adjust as necessary
    
    # Helper function to calculate metrics and format them
    calculateAndFormatMetrics <- function(returns, benchmark_returns = NULL) {
      total_return <- Return.cumulative(returns, geometric = TRUE)
      annual_return <- mean(Return.annualized(returns, scale = 252))
      annual_volatility <- sd(returns) * sqrt(252)
      sharpe_ratio <- (annual_return-risk_free_rate)/annual_volatility
      max_drawdown <- -maxDrawdown(returns)
      beta <- if (!is.null(benchmark_returns)) CAPM.beta(returns, benchmark_returns, Rf = risk_free_rate) else NA
      
      # Format all the metrics to round to 4 decimal places
      c(
        Total_Return = round(total_return, 4),
        Annual_Return = round(annual_return, 4),
        Annual_Volatility = round(annual_volatility, 4),
        Sharpe_Ratio = round(sharpe_ratio, 4),
        Max_Drawdown = round(max_drawdown, 4),
        Beta = round(beta, 4)
      )
    }
    
    # Calculate metrics for fund and benchmark
    fund <- calculateAndFormatMetrics(fund_returns, benchmark_returns)
    benchmark <- calculateAndFormatMetrics(benchmark_returns)
    
    # Create a matrix to bind the metrics into a data frame with row names as the first column
    metrics_data <- rbind(fund, benchmark)
    metrics_df <- data.frame(Metric = rownames(metrics_data), metrics_data)
    rownames(metrics_df) <- NULL
    
    # Find the index of the row with the highest Sharpe Ratio
    best_sharpe_index <- which.max(metrics_df$Sharpe_Ratio)
    
    DT::datatable(metrics_df, options = list(
      pageLength = 1,
      searching = FALSE,
      lengthChange = FALSE,
      paging = FALSE, 
      info = FALSE 
    ), rownames = FALSE) %>%
      DT::formatStyle(
        'Metric',
        target = 'row', 
        backgroundColor = styleEqual(metrics_df$Sharpe_Ratio, rep(c('white', '#EBF5FB'), length.out = length(metrics_df$Sharpe_Ratio))[best_sharpe_index]),
        color = styleEqual(metrics_df$Sharpe_Ratio, rep(c('black', 'navy'), length.out = length(metrics_df$Sharpe_Ratio))[best_sharpe_index])
      ) %>%
      DT::formatStyle(
        columns = c('Total_Return', 'Annual_Return', 'Annual_Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'Beta'),
        fontWeight = styleEqual(best_sharpe_index - 1, 'bold') 
      ) 
  }, server = FALSE)
  
  # Once the app is initialized, trigger the data fetch
  observe({
    shinyjs::runjs('$("#runAnalysis").click();')
  })
}

shinyApp(ui = ui, server = server)

