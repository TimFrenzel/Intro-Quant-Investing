
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from quarterlyRebalancePortfolio import quarterlyRebalanceReturnsFromFile
from fetchDataAndMakeDFs import getQuarter
from fetchDataAndMakeDFs import calcBeta
from fetchDataAndMakeDFs import convert_to_datetime
#from oldFilesFolder.quarterlyRebalanceFromFile import normalizePricesFromFile

def spliceMarketDF(marketDF, startDate, endDate):
    marketDates = marketDF["Dates"].tolist()
    marketDatesObjects = convert_to_datetime(marketDates)
    marketValues = marketDF["SPX Index"].tolist()

    market_start_index = marketDatesObjects.index(startDate)
    market_end_index = marketDatesObjects.index(endDate)
    market_spliced_values = marketValues[market_start_index:market_end_index+1]
    market_spliced_dates = marketDates[market_start_index:market_end_index+1]
    dataDict = {"Dates": market_spliced_dates, "SPX Index": market_spliced_values}
    splicedMarketDF = pd.DataFrame(dataDict)
    
    return splicedMarketDF



def calcFactorBeta(portfolioDF, marketDF):
    portfolioDates = portfolioDF["Dates"].tolist()
    portfolioValues = portfolioDF["Portfolio"].tolist()
    
    marketDates = marketDF["Dates"].tolist()
    marketDatesObjects = convert_to_datetime(marketDates)
    marketValues = marketDF["SPX Index"].tolist()
    
    startDate = portfolioDates[-127]
    endDate = portfolioDates[-1]


    portfolio_start_index = portfolioDates.index(startDate)
    portfolio_end_index = portfolioDates.index(endDate)
    portfolio_spliced_Values = portfolioValues[portfolio_start_index:portfolio_end_index+1]

    market_start_index = marketDatesObjects.index(startDate)
    market_end_index = marketDatesObjects.index(endDate)
    market_spliced_values = marketValues[market_start_index:market_end_index+1]

    
    beta = calcBeta(portfolio_spliced_Values, market_spliced_values)
    return beta
    
    


def newSharpeRatio(PortfolioDF, riskFreeRate):
    returns = calcPortfolioReturns(PortfolioDF)
    returns['Dates'] = pd.to_datetime(returns['Dates'])  # Convert 'Date' column to datetime if needed

    monthly_returns = np.log(1 + returns.groupby(pd.Grouper(key='Dates', freq='ME')).sum())
    combined_returns = monthly_returns
    portfolio_monthly_returns = combined_returns.mean(axis=1)
    annual_average_return = portfolio_monthly_returns.mean() * 12
    annual_std_dev = portfolio_monthly_returns.std() * np.sqrt(12)

    # Calculate Sharpe ratio
    sharpe_ratio = (annual_average_return - riskFreeRate) / annual_std_dev

    return sharpe_ratio


    
    
def calcPortfolioReturns(PortfolioDF):
        # Extract prices from the DataFrame
    if 'Portfolio' in PortfolioDF.columns:
        price_values = PortfolioDF['Portfolio'].values
    else:
        price_values = PortfolioDF['SPX Index'].values
        
    date_values = PortfolioDF['Dates']
    #print(PortfolioDF)
    returns = []
    dates = []
    for i in range(1, len(price_values)):
        #daily_return = (price_values[i] - price_values[i-1]) / price_values[i-1]  # calculate daily return
        #p
        log_return = np.log(price_values[i] / price_values[i-1])

        returns.append(log_return)
        date = date_values[i]
        #print(date)
        dates.append(date)

    # Create a new DataFrame for returns
    #print(i)
    returns_df = pd.DataFrame({'Daily Return': returns, 'Dates': dates})
    
    return returns_df
    
    


def graphPriceData(portfolioDF, marketDF):
    dates2 = portfolioDF["Dates"].tolist()
    prices1 = portfolioDF["Portfolio"].tolist()
    prices2 = marketDF["SPX Index"].tolist()
    dates = [dt.strftime("%m/%d/%Y") for dt in dates2]
    
    numDates = len(dates)

    intervals = numDates / 12
    ticks = int(intervals)

    plt.figure(figsize=(10, 6))

    plt.plot(dates, prices1, label="Portfolio", color='blue')
    plt.plot(dates, prices2, label="S&P", color='green')

    plt.xticks(dates[::ticks], rotation=45)  # Display every 10th date


    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Portfolio vs S&P')
    plt.legend()
    plt.grid(True)
    plt.show()



def graphPriceData2(portfolioDF, marketDF):
    dates2 = portfolioDF["Dates"].tolist()
    prices1 = portfolioDF["Portfolio"].tolist()
    prices2 = marketDF["SPX Index"].tolist()
    dates = [dt.strftime("%m/%d/%Y") for dt in dates2]
    
    numDates = len(dates)

    intervals = numDates / 12
    ticks = int(intervals)

    plt.figure(figsize=(10, 6))

    plt.plot(dates, prices1, label="Portfolio", color='blue')
    plt.plot(dates, prices2, label="S&P", color='green')

    plt.xticks(dates[::ticks], rotation=45)  # Display every 10th date


    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Portfolio vs S&P')
    plt.legend()
    plt.grid(True)
    plt.show()



    
    
def calcYearlyReturn(stockDF, years):
    
    if 'Portfolio' in stockDF.columns:
        values_list = stockDF['Portfolio'].values
    else:
        values_list = stockDF['SPX Index'].values
    

    startValue = values_list[0]
    #print("start Value")
    #print(startValue)
    #print(startValue)
    #print("endValue")
    endValue = values_list[-1]
    #print(endValue)
    yearlyReturn = ((((float(endValue) / float(startValue))**(1/years)) -1))
    
    #print("yearly return")
    #print(yearlyReturn)
    return yearlyReturn


    
def calcMaximumDrawdown(stockDF):

    if 'Portfolio' in stockDF.columns:
        price_values = stockDF['Portfolio'].values
    else:
        price_values = stockDF['SPX Index'].values

    
    max_drawdown = 0
    peak_value = price_values[0]  # Initial peak value
    trough_value = peak_value  # Initial trough value

    for price in price_values:
        # Update peak value if current price is greater than previous peak
        if price > peak_value:
            peak_value = price
            trough_value = price  # Reset trough value
            
        # Update trough value if current price is lower than trough value
        elif price < trough_value:
            trough_value = price
            
            # Calculate drawdown and update max_drawdown if necessary
            drawdown = (peak_value - price) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown
    
def calcVolatility(stockDF):
    
    if 'Portfolio' in stockDF.columns:
        price_values = stockDF['Portfolio'].values
    else:
        price_values = stockDF['SPX Index'].values
    
    date_values = stockDF['Dates'].tolist()

    returns = []
    dates = []
    for i in range(1, len(price_values)):
        daily_return = (price_values[i] - price_values[i-1]) / price_values[i-1]  # calculate daily return
        daily_return += 1
        daily_return = np.around(daily_return, 6)
        #print(daily_return)
        returns.append(daily_return)
        date = date_values[i]
        dates.append(date)

    # Create a new DataFrame for returns
    returns_df = pd.DataFrame({'Daily Return': returns, 'Dates': dates})
    

    daily_volatility = returns_df['Daily Return'].std(ddof=0)
    #print(daily_volatility)
    #print('here')
    #daily_volatility = price_values.std(ddof=0)

    annualized_volatility = np.sqrt(252 * 1) * daily_volatility # took out years

    return annualized_volatility
    

    
    


# def createReturnsDFforBeta(stockDF):
#     price_values = stockDF['Portfolio'].values
#     date_values = stockDF['Dates']

#     returns = []
#     dates = []
#     for i in range(1, len(price_values)):
#         daily_return = (price_values[i] - price_values[i-1]) / price_values[i-1]  # calculate daily return
#         #print(i)
#         #print(price_values[i])
#         #print("price at i")
#         #print(price_values[i])
#         #daily_return += 1
#         daily_return = np.around(daily_return, 6)
#         #print(daily_return)
#         returns.append(daily_return)
#         date = date_values[i]
#         dates.append(date)

#     # Create a new DataFrame for returns
#     returns_df = pd.DataFrame({'Daily Return': returns, 'Dates': dates})
#     return returns_df




# def calcBeta(stockDF, marketDF):
#     stock_returns = createReturnsDFforBeta(stockDF)
#     market_returns = createReturnsDFforBeta(marketDF)
    
#     covariance = np.cov(stock_returns['Daily Return'], market_returns['Daily Return'])[0, 1]
#     market_variance = np.var(market_returns['Daily Return'])
#     beta = covariance / market_variance
#     return beta

        
    
        
    
def createPortfolioToCalculateTurnover(percentageList, stockList, dates, initialInvestment, stockDailyReturnDFs):
    stockDict = {}
    for i in range(len(stockList)):
        var_Name = "list_" + str(i)
        list_of_stock_prices = []
        initialValue = initialInvestment * percentageList[i]
        list_of_stock_prices.append(initialValue)
        stockDict[var_Name] = list_of_stock_prices
    
    turnoverList = []
        
    totalTurnover = quarterlyRebalanceToCalculateTurnover(stockDict, percentageList, dates, stockDailyReturnDFs, turnoverList)
    return totalTurnover
    
def quarterlyRebalanceToCalculateTurnover(stockDict, percentageList, dates, normalizedDFs, turnoverList):


    for i in range(0, len(stockDict)):
        #print(returnsDF)
        var_Name = "list_" + str(i)
        #if(len(stockDict[var_Name]) == 0):  
        totalPortfolioValue = 0
            
        stockList = stockDict[var_Name]

        j = len(stockDict[var_Name]) -1 #-1 for being dict having an initial value
        current_date = dates[j]
        if(j+1 < len(dates)):
            next_date = dates[j+1] #1
            #print("check")
        else:
            next_date = current_date
        #print(stockDict)  
        #print(j) 
        x = 0
        while(getQuarter(current_date) == getQuarter(next_date) and x == 0):
            stockList = stockDict[var_Name]
            stock_price = stockList[-1] * normalizedDFs[i]['Daily Return'][j]
            stockList.append(stock_price)
            #print("size of j")
            j = len(stockList) -1
            
            x = 0
            if(dates[j] == dates[-1]):
                x = 1
            if(x == 0):
                next_date = dates[j+1] #2
                #stockList.append(stock_price)
            #print(len(dates))

            #j+=1

    totalPortfolioValue = 0  
    for i in range(0,len(stockDict)):
        var_Name = "list_" + str(i)
        stockList = stockDict[var_Name]
        lastPrice = stockList[-1] 
        totalPortfolioValue += lastPrice
     
    if(x == 0):   
        for i in range(0, len(stockDict)):
            var_Name = "list_" + str(i)
            stockList = stockDict[var_Name]
            endQuarterPrice = totalPortfolioValue * percentageList[i]


            stockList[-1] = endQuarterPrice
            j = len(stockList) -1
            startMonthPrice = stockList[-1] * normalizedDFs[i]['Daily Return'][j]
            stockList.append(startMonthPrice)
            
            if(endQuarterPrice > startMonthPrice):
                turnover = endQuarterPrice - startMonthPrice
                turnoverList.append(turnover)

        if(len(dates) != len(stockDict[var_Name])):
            quarterlyRebalanceToCalculateTurnover(stockDict, percentageList, dates, normalizedDFs, turnoverList)
    
    totalTurnover = sum(turnoverList)
    return turnoverList
    #return totalTurnover    
    
def plot_bar_graph(years, values):
    plt.figure(figsize=(10, 6))
    plt.bar(years, values, color='skyblue')
    plt.xlabel('Years')
    plt.ylabel('Alpha')
    plt.title('Factor Alpha vs S&P Alpha')
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
    
    
    
    
def printVals(i1, i2, i3, i4):
    print(i1)
    print(i2)
    print(i3)
    print(i4)
    
    return i3