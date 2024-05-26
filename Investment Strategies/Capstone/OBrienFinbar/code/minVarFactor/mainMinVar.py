from datetime import datetime
import numpy as np
from fetchDataAndMakeDFs import process_input_file
from fetchDataAndMakeDFs import determine_date_format
from fetchDataAndMakeDFs import pricesToDailyReturns
from fetchDataAndMakeDFs import convert_to_datetime
from fetchDataAndMakeDFs import convert_to_datetime2
from fetchDataAndMakeDFs import convert_to_datetime3

from fetchDataAndMakeDFs import parse_csv_to_dataframe
from fetchDataAndMakeDFs import createDailyReturnDF
from fetchDataAndMakeDFs import remove_additional_commas
from fetchDataAndMakeDFs import makeDailyBetaDfs
from fetchDataAndMakeDFs import getQuarterDates
from MinVarFactor import getLongAndShortBetaPositions
from MinVarFactor import minVarFactorPorfolio
from MinVarFactor import minVarFactorPorfolioLong
from MinVarFactor import minVarFactorPorfolioShort

from minVarComputingFunctions import spliceMarketDF
import pandas as pd
from minVarComputingFunctions import newSharpeRatio
from minVarComputingFunctions import calcMaximumDrawdown
from minVarComputingFunctions import calcVolatility
from minVarComputingFunctions import calcYearlyReturn
from minVarComputingFunctions import calcFactorBeta
from minVarComputingFunctions import graphPriceData
from minVarComputingFunctions import graphPriceData2

from fetchDataAndMakeDFs import normalizePrice
import matplotlib.pyplot as plt
from minVarComputingFunctions import plot_bar_graph

def runFactorPortfolio(startDate, endDate, numBetaLookbackDays, numBetaGroupingDays):


    fileData = parse_csv_to_dataframe("test3.csv")
    stockList = []

    for key in fileData.keys():
        stockList.append(key)

    # print(len(stockList))

    dailyReturnsDf, dailyReturnsDates = createDailyReturnDF(fileData, stockList, "a", startDate,endDate) #"6/1/2012","3/6/2023", 
    print("made returns df")
    # print(dailyReturnsDf)














    benchMarkData = parse_csv_to_dataframe("MarketData.csv")
    print("got benchmark data")

    benchMarkList = []

    for key in benchMarkData.keys():
        benchMarkList.append(key)



    benchmarkReturnsDf, benchMarkReturnsDates = createDailyReturnDF(benchMarkData, benchMarkList, "a", startDate,endDate) #1/3/2023, "6/1/2012","3/6/2023"
    print("made benchmark returns data")


    dates = dailyReturnsDf["Dates"].tolist()


    print("making betas dfs")
    
    
    
    if numBetaLookbackDays == 252:
        if numBetaGroupingDays == 28:
            dailyBetasDf = pd.read_csv('lookback1YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback1YearGroupingDaily.csv')
    elif numBetaLookbackDays == 756:
        if numBetaGroupingDays == 28:
            dailyBetasDf = pd.read_csv('lookback3YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback3YearGroupingDaily.csv')
    else:
        if numBetaGroupingDays == 28:
            dailyBetasDf = pd.read_csv('lookback.5YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback.5YearGroupingDaily.csv')
        
    
    
    
    
    
    

    # dailyBetasDF, betaDates = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)
    # dailyBetasDF = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)

    # dailyBetasDF.to_csv('lookback1YearGroupingMonthly.csv', index=False)

    
    
    betaDates = dailyBetasDF['Dates'].tolist()
    # print(type(betaDates))
    print("here is daily betas df")


    betaQuarterChangesDates = getQuarterDates(betaDates)
    print(betaQuarterChangesDates)

    high_beta_names, low_beta_names = getLongAndShortBetaPositions(betaQuarterChangesDates, dailyBetasDF)
    print("made high/low beta names")



    min_var_portfolios = minVarFactorPorfolio(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)
    print("here is our combined portfolio")
    print(min_var_portfolios)
    combined_min_var_dataframe = pd.concat(min_var_portfolios, ignore_index= True)
    print(combined_min_var_dataframe)



    factorDatesList = combined_min_var_dataframe["Dates"].tolist()
    factorStartDate = factorDatesList[0]
    factorEndDate = factorDatesList[-1]
    splicedMarketDF = spliceMarketDF(benchMarkData, factorStartDate, factorEndDate)
    marketDFPriceDF = normalizePrice(splicedMarketDF)
    print(marketDFPriceDF)

    print("DATE RANGE")
    print(factorStartDate)
    print(factorEndDate)
    print(" ")

    total_sharpe = newSharpeRatio(combined_min_var_dataframe, 0)
    print("ALL VALUES FOR FACTOR PORTFOLIO")
    print("sharpe ratio")
    print(total_sharpe)
    print(" ")

    maxdd = calcMaximumDrawdown(combined_min_var_dataframe)
    print("Max drawdown")
    print(maxdd)
    print(" ")

    volatility = calcVolatility(combined_min_var_dataframe)
    print("annual volatility")
    print(volatility)
    print(" ")


    numYears = len(factorDatesList) / 252

    yearlyReturn = calcYearlyReturn(combined_min_var_dataframe, numYears)
    print("yearly return")
    print(yearlyReturn)
    print(" ")


    portfolio_daily_returns = pricesToDailyReturns(combined_min_var_dataframe, ["Portfolio"])

    beta = calcFactorBeta(portfolio_daily_returns, benchmarkReturnsDf)
    print("beta")
    print(beta)
    print(" ")

    print("ALL VALUES FOR BENCHMARK")
    BM_sharpe = newSharpeRatio(splicedMarketDF, 0)
    print("sharpe ratio")
    print(BM_sharpe)
    print(" ")

    BM_maxdd = calcMaximumDrawdown(splicedMarketDF)
    print("max DD")
    print(BM_maxdd)
    print(" ")

    BM_volatility = calcVolatility(splicedMarketDF)
    print("volatility")
    print(BM_volatility)
    print(" ")

    BM_yearly_return = calcYearlyReturn(splicedMarketDF, numYears)
    print("Yearly return")
    print(BM_yearly_return)
    print(" ")

    alphaDifference = yearlyReturn - BM_yearly_return

    print("ALPHA DIFFERENCE")
    print(alphaDifference)


    graphPriceData(combined_min_var_dataframe, marketDFPriceDF)

# runFactorPortfolio("6/1/2012","3/6/2023", 252, 28)
# runFactorPortfolio("1/26/2021","3/6/2023", 252, 28)
# runFactorPortfolio("12/31/2009","3/6/2023", 252, 28)
























# def runFactorPortfolio2(startDate, endDate, numBetaLookbackDays, numBetaGroupingDays):


#     fileData = parse_csv_to_dataframe("test3.csv")
#     stockList = []

#     for key in fileData.keys():
#         stockList.append(key)

#     # print(len(stockList))

#     dailyReturnsDf, dailyReturnsDates = createDailyReturnDF(fileData, stockList, "a", startDate,endDate) #"6/1/2012","3/6/2023", 
#     print("made returns df")
#     # print(dailyReturnsDf)








#     benchMarkData = parse_csv_to_dataframe("MarketData.csv")
#     print("got benchmark data")

#     benchMarkList = []

#     for key in benchMarkData.keys():
#         benchMarkList.append(key)



#     benchmarkReturnsDf, benchMarkReturnsDates = createDailyReturnDF(benchMarkData, benchMarkList, "a", startDate,endDate) #1/3/2023, "6/1/2012","3/6/2023"
#     print("made benchmark returns data")


#     dates = dailyReturnsDf["Dates"].tolist()


#     print("making betas dfs")

#     # dailyBetasDF, betaDates = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)
    
    
    
    
#     # dailyBetasDF = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)

#     # dailyBetasDF.to_csv('lookback3YearGroupingMonthly.csv', index=False)
#     dailyBetasDF = pd.read_csv('lookback1YearGroupingMonthly.csv')
#     # print(dailyBetasDF)



    
    
#     betaDates = dailyBetasDF['Dates'].tolist()
#     betaDates = convert_to_datetime2(betaDates)
#     dailyBetasDF["Dates"] = betaDates
#     # print(type(betaDates))
#     print("here is daily betas df")


#     betaQuarterChangesDates = getQuarterDates(betaDates)
#     # print(betaQuarterChangesDates)
    
    
    
#     # z = dailyBetasDF.loc[dailyBetasDF["Dates"] == "2011-03-31"].index[0]
#     # print(dailyBetasDF["Dates"])
    
    
    
    
    

#     high_beta_names, low_beta_names = getLongAndShortBetaPositions(betaQuarterChangesDates, dailyBetasDF)
#     print("made high/low beta names")



#     min_var_portfolios = minVarFactorPorfolio(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)
#     print("here is our combined portfolio")
#     print(min_var_portfolios)
#     combined_min_var_dataframe = pd.concat(min_var_portfolios, ignore_index= True)
#     print(combined_min_var_dataframe)



#     factorDatesList = combined_min_var_dataframe["Dates"].tolist()
#     factorStartDate = factorDatesList[0]
#     factorEndDate = factorDatesList[-1]
#     splicedMarketDF = spliceMarketDF(benchMarkData, factorStartDate, factorEndDate)
#     marketDFPriceDF = normalizePrice(splicedMarketDF)
#     print(marketDFPriceDF)

#     print("DATE RANGE")
#     print(factorStartDate)
#     print(factorEndDate)
#     print(" ")

#     total_sharpe = newSharpeRatio(combined_min_var_dataframe, 0)
#     print("ALL VALUES FOR FACTOR PORTFOLIO")
#     print("sharpe ratio")
#     print(total_sharpe)
#     print(" ")

#     maxdd = calcMaximumDrawdown(combined_min_var_dataframe)
#     print("Max drawdown")
#     print(maxdd)
#     print(" ")

#     volatility = calcVolatility(combined_min_var_dataframe)
#     print("annual volatility")
#     print(volatility)
#     print(" ")


#     numYears = len(factorDatesList) / 252

#     yearlyReturn = calcYearlyReturn(combined_min_var_dataframe, numYears)
#     print("yearly return")
#     print(yearlyReturn)
#     print(" ")


#     portfolio_daily_returns = pricesToDailyReturns(combined_min_var_dataframe, ["Portfolio"])

#     beta = calcFactorBeta(portfolio_daily_returns, benchmarkReturnsDf)
#     print("beta")
#     print(beta)
#     print(" ")

#     print("ALL VALUES FOR BENCHMARK")
#     BM_sharpe = newSharpeRatio(splicedMarketDF, 0)
#     print("sharpe ratio")
#     print(BM_sharpe)
#     print(" ")

#     BM_maxdd = calcMaximumDrawdown(splicedMarketDF)
#     print("max DD")
#     print(BM_maxdd)
#     print(" ")

#     BM_volatility = calcVolatility(splicedMarketDF)
#     print("volatility")
#     print(BM_volatility)
#     print(" ")

#     BM_yearly_return = calcYearlyReturn(splicedMarketDF, numYears)
#     print("Yearly return")
#     print(BM_yearly_return)
#     print(" ")

#     alphaDifference = yearlyReturn - BM_yearly_return

#     print("ALPHA DIFFERENCE")
#     print(alphaDifference)


#     graphPriceData(combined_min_var_dataframe, marketDFPriceDF)

# runFactorPortfolio("6/1/2012","3/6/2023", 252, 28)
# runFactorPortfolio("1/26/2021","3/6/2023", 252, 28)
# runFactorPortfolio("12/31/2009","3/6/2023", 252, 28)

# 12/31/2009

















def runFactorPortfolio3(startDate, endDate, numBetaLookbackDays, numBetaGroupingDays, investmentStrategy='C'):
    print("running...")
    # print(endDate)
    original_startdate_obj = datetime.fromisoformat(startDate)
    print("running...")
    # Format the datetime object as a string in the desired format
    startDate = original_startdate_obj.strftime("%m/%d/%Y")
    
    original_enddate_obj = datetime.fromisoformat(endDate)

    # Format the datetime object as a string in the desired format
    endDate = original_enddate_obj.strftime("%m/%d/%Y")
    # print(startDate)
    if startDate[-7] == '0':
        startDate = startDate[:-7] + startDate[-7 + 1:]
        
    if endDate[-7] == '0':
        endDate = endDate[:-7] + endDate[-7 + 1:]

    
    if(startDate[0] == '0'):
        startDate = startDate[1:]
    # print(startDate)
    
    if(endDate[0] == '0'):
        endDate = endDate[1:]
    # print(endDate)
    print("running...")

    fileData = parse_csv_to_dataframe("test3.csv")
    stockList = []

    for key in fileData.keys():
        stockList.append(key)

    # print(len(stockList))
    
    # fileData['Dates'] = convert_to_datetime3(fileData['Dates'])

    dailyReturnsDf, dailyReturnsDates = createDailyReturnDF(fileData, stockList, "a", startDate,endDate) #"6/1/2012","3/6/2023", 
    # print("made returns df")
    # print(dailyReturnsDf)

    # print(dailyReturnsDf)
    print("returns df")
    print(dailyReturnsDf)





    benchMarkData = parse_csv_to_dataframe("MarketData.csv")
    # print("got benchmark data")

    benchMarkList = []

    for key in benchMarkData.keys():
        benchMarkList.append(key)



    benchmarkReturnsDf, benchMarkReturnsDates = createDailyReturnDF(benchMarkData, benchMarkList, "a", startDate,endDate) #1/3/2023, "6/1/2012","3/6/2023"
    # print("made benchmark returns data")
    print("market returns")
    print(benchmarkReturnsDf)

    dates = dailyReturnsDf["Dates"].tolist()


    print("making betas dfs")

    # dailyBetasDF, betaDates = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)
    
    # dailyBetasDF, betaDates = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)

    
    
    # dailyBetasDF = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates, numBetaLookbackDays, numBetaGroupingDays)

    # dailyBetasDF.to_csv('lookback3YearGroupingDaily.csv', index=False)
    # dailyBetasDF = pd.read_csv('lookback1YearGroupingMonthly.csv')
    # print(dailyBetasDF)
    
    # print(numBetaLookbackDays)
    # print(numBetaGroupingDays)

    if numBetaLookbackDays == 252:
        if numBetaGroupingDays == 28:
            dailyBetasDF = pd.read_csv('lookback1YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback1YearGroupingDaily.csv')
    elif numBetaLookbackDays == 756:
        if numBetaGroupingDays == 28:
            dailyBetasDF = pd.read_csv('lookback3YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback3YearGroupingDaily.csv')
    else:
        if numBetaGroupingDays == 28:
            dailyBetasDF = pd.read_csv('lookback.5YearGroupingMonthly.csv')
        else:
            dailyBetasDF = pd.read_csv('lookback.5YearGroupingDaily.csv')


    print("pulled beta csv")
    
    betaDates = dailyBetasDF['Dates'].tolist()
    # print(betaDates)
    # betaDateStartIndex = betaDates.index(startDate)
    # betaDateEndIndex = betaDates.index(endDate)
    
    # betaDates = betaDates[betaDateStartIndex:betaDateEndIndex+1]
    


    betaDates = convert_to_datetime2(betaDates)
    print('convetred beta dates')
    # print(betaDates)
    
    # print(startDate)
    # print(endDate)
    
    
    # # startDate = convert_to_datetime(startDate)
    # # endDate = convert_to_datetime(endDate)
    
    
    startDate = datetime.strptime(startDate, "%m/%d/%Y")
    endDate = datetime.strptime(endDate, "%m/%d/%Y")

    
    
    
    
    print(betaDates[:75]) 
    print(startDate)
    print(endDate)
    print("l")

    
    
    
    # betaDateStartIndex = betaDates.index(startDate)
    betaDateEndIndex = betaDates.index(endDate)
    if startDate in betaDates:
        betaDateStartIndex = betaDates.index(startDate)

        betaDates = betaDates[betaDateStartIndex:betaDateEndIndex+1]
    else:
        betaDates = betaDates[:betaDateEndIndex+1]

    
    
    
    
    
    dailyBetasDF["Dates"] = convert_to_datetime2(dailyBetasDF["Dates"])
    # dfStartIndex = dailyBetasDF["Dates"][dailyBetasDF["Dates"] == startDate].index[0]
    # dfEndIndex = dailyBetasDF["Dates"][dailyBetasDF["Dates"] == endDate].index[0]
    # dailyBetasDF = dailyBetasDF.iloc[dfStartIndex: dfEndIndex +1]

    # print(type(betaDates))
    # print("here is daily betas df")


    betaQuarterChangesDates = getQuarterDates(betaDates)
    # print(betaQuarterChangesDates)
    
    print("got q change dates")
    
    # z = dailyBetasDF.loc[dailyBetasDF["Dates"] == "2011-03-31"].index[0]
    # print(dailyBetasDF["Dates"])
    
    print("check")
    print(betaQuarterChangesDates)
    print(dailyBetasDF)
    

    high_beta_names, low_beta_names = getLongAndShortBetaPositions(betaQuarterChangesDates, dailyBetasDF)
    print("made high/low beta names")

    # min_var_portfolios = minVarFactorPorfolio(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)

    
    print("betaDates")
    print(betaDates)
    
    print("quarter change dates")
    print(betaQuarterChangesDates)
    
    
    
    
    if investmentStrategy == 'L':
        min_var_portfolios = minVarFactorPorfolioLong(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)
    elif investmentStrategy == 'S':
        min_var_portfolios = minVarFactorPorfolioShort(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)
    else:
        min_var_portfolios = minVarFactorPorfolio(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)

    # print("here is our combined portfolio")
    # print(min_var_portfolios)
    print("made minVar portfolios")
    combined_min_var_dataframe = pd.concat(min_var_portfolios, ignore_index= True)
    # print(combined_min_var_dataframe)

    print("Combined portfolios...")
    print(combined_min_var_dataframe)


    factorDatesList = combined_min_var_dataframe["Dates"].tolist()
    factorStartDate = factorDatesList[0]
    factorEndDate = factorDatesList[-1]
    splicedMarketDF = spliceMarketDF(benchMarkData, factorStartDate, factorEndDate)
    marketDFPriceDF = normalizePrice(splicedMarketDF)
    # print(marketDFPriceDF)

    # print("DATE RANGE")
    # print(factorStartDate)
    # print(factorEndDate)
    # print(" ")

    total_sharpe = newSharpeRatio(combined_min_var_dataframe, 0)
    # print("ALL VALUES FOR FACTOR PORTFOLIO")
    # print("sharpe ratio")
    # print(total_sharpe)
    # print(" ")

    maxdd = calcMaximumDrawdown(combined_min_var_dataframe)
    # print("Max drawdown")
    # print(maxdd)
    # print(" ")

    volatility = calcVolatility(combined_min_var_dataframe)
    # print("annual volatility")
    # print(volatility)
    # print(" ")


    numYears = len(factorDatesList) / 252

    yearlyReturn = calcYearlyReturn(combined_min_var_dataframe, numYears)
    # print("yearly return")
    # print(yearlyReturn)
    # print(" ")


    portfolio_daily_returns = pricesToDailyReturns(combined_min_var_dataframe, ["Portfolio"])

    beta = calcFactorBeta(portfolio_daily_returns, benchmarkReturnsDf)
    # print("beta")
    # print(beta)
    # print(" ")

    # print("ALL VALUES FOR BENCHMARK")
    BM_sharpe = newSharpeRatio(splicedMarketDF, 0)
    # print("sharpe ratio")
    # print(BM_sharpe)
    # print(" ")

    BM_maxdd = calcMaximumDrawdown(splicedMarketDF)
    # print("max DD")
    # print(BM_maxdd)
    # print(" ")

    BM_volatility = calcVolatility(splicedMarketDF)
    # print("volatility")
    # print(BM_volatility)
    # print(" ")

    BM_yearly_return = calcYearlyReturn(splicedMarketDF, numYears)
    # print("Yearly return")
    # print(BM_yearly_return)
    # print(" ")

    alphaDifference = yearlyReturn - BM_yearly_return

    # print("ALPHA DIFFERENCE")
    # print(alphaDifference)


    # graphPriceData(combined_min_var_dataframe, marketDFPriceDF)
    listOfStuffToPrint = ["DATE RANGE", factorStartDate, factorEndDate, "ALL VALUES FOR FACTOR PORTFOLIO", "sharpe ratio", total_sharpe, "Max drawdown", maxdd, "annual volatility", volatility, "yearly return", yearlyReturn, "beta", beta, "ALL VALUES FOR BENCHMARK", "sharpe ratio", BM_sharpe, "max DD", BM_maxdd, "volatility", BM_volatility, "Yearly return", BM_yearly_return,"ALPHA DIFFERENCE", alphaDifference]
    
    # a = listOfStuffToPrint[::2]
    # b = listOfStuffToPrint[1::2]
    
    ourPortfolio = [total_sharpe, maxdd, volatility,yearlyReturn, beta, yearlyReturn - BM_yearly_return]
    bm = [BM_sharpe, BM_maxdd, BM_volatility, BM_yearly_return, 1, BM_yearly_return - yearlyReturn]
    measures = ["Sharpe Ratio", "Max Drawdown", "Volatility", "Yearly Return", "Beta", "Alpha"]
    
    # print(len(a))
    # print(len(b))
    # for i in range(len(a)):
    #     print(a[i])
    #     print(b[i])
    
    outputDF = pd.DataFrame({
    'Measures': measures,
    'Our Portfolio': ourPortfolio, 
    'Benchmark': bm    
})
    # print(df)
    
    portfolioXVals = combined_min_var_dataframe['Dates'].tolist()
    portfolioYVals = combined_min_var_dataframe['Portfolio'].tolist()

    marketXVals = marketDFPriceDF['Dates'].tolist()
    marketYVals = marketDFPriceDF['SPX Index'].tolist()

    # print(marketDFPriceDF)
    # for i in range(len(marketXVals) // 4):
    #     print(f'D: {portfolioXVals[i]}. P:{portfolioYVals[i]}')
    # graphPriceData(combined_min_var_dataframe, marketDFPriceDF)
    
    print(combined_min_var_dataframe)

    
    return outputDF, portfolioXVals, portfolioYVals, marketXVals, marketYVals
    
    
    
    # return outputDF, 1,2,3,4
    



# runFactorPortfolio3("12/31/2009","3/6/2023", 252, 28)











































# runFactorPortfolio("6/1/2012","3/6/2023", 252, 28)
    
    
    
    
    

# fileData = parse_csv_to_dataframe("test3.csv")
# stockList = []

# for key in fileData.keys():
#     stockList.append(key)

# # print(len(stockList))

# dailyReturnsDf, dailyReturnsDates = createDailyReturnDF(fileData, stockList, "a", "6/1/2012","3/6/2023") #"6/1/2012","3/6/2023", 
# print("made returns df")
# # print(dailyReturnsDf)








# benchMarkData = parse_csv_to_dataframe("MarketData.csv")
# print("got benchmark data")

# benchMarkList = []

# for key in benchMarkData.keys():
#     benchMarkList.append(key)
    


# benchmarkReturnsDf, benchMarkReturnsDates = createDailyReturnDF(benchMarkData, benchMarkList, "a", "6/1/2012","3/6/2023") #1/3/2023, "6/1/2012","3/6/2023"
# print("made benchmark returns data")


# dates = dailyReturnsDf["Dates"].tolist()


# print("making betas dfs")

# dailyBetasDF, betaDates = makeDailyBetaDfs(dailyReturnsDf, benchmarkReturnsDf, stockList, dates)
# print("here is daily betas df")


# betaQuarterChangesDates = getQuarterDates(betaDates)

    
# high_beta_names, low_beta_names = getLongAndShortBetaPositions(betaQuarterChangesDates, dailyBetasDF)
# print("made high/low beta names")



# min_var_portfolios = minVarFactorPorfolio(high_beta_names, low_beta_names, betaQuarterChangesDates, dailyReturnsDf, stockList, betaDates)
# print("here is our combined portfolio")
# print(min_var_portfolios)
# combined_min_var_dataframe = pd.concat(min_var_portfolios, ignore_index= True)
# print(combined_min_var_dataframe)



# factorDatesList = combined_min_var_dataframe["Dates"].tolist()
# factorStartDate = factorDatesList[0]
# factorEndDate = factorDatesList[-1]
# splicedMarketDF = spliceMarketDF(benchMarkData, factorStartDate, factorEndDate)
# marketDFPriceDF = normalizePrice(splicedMarketDF)
# print(marketDFPriceDF)

# print("DATE RANGE")
# print(factorStartDate)
# print(factorEndDate)
# print(" ")

# total_sharpe = newSharpeRatio(combined_min_var_dataframe, 0)
# print("ALL VALUES FOR FACTOR PORTFOLIO")
# print("sharpe ratio")
# print(total_sharpe)
# print(" ")

# maxdd = calcMaximumDrawdown(combined_min_var_dataframe)
# print("Max drawdown")
# print(maxdd)
# print(" ")

# volatility = calcVolatility(combined_min_var_dataframe)
# print("annual volatility")
# print(volatility)
# print(" ")


# numYears = len(factorDatesList) / 252

# yearlyReturn = calcYearlyReturn(combined_min_var_dataframe, numYears)
# print("yearly return")
# print(yearlyReturn)
# print(" ")


# portfolio_daily_returns = pricesToDailyReturns(combined_min_var_dataframe, ["Portfolio"])

# beta = calcFactorBeta(portfolio_daily_returns, benchmarkReturnsDf)
# print("beta")
# print(beta)
# print(" ")

# print("ALL VALUES FOR BENCHMARK")
# BM_sharpe = newSharpeRatio(splicedMarketDF, 0)
# print("sharpe ratio")
# print(BM_sharpe)
# print(" ")

# BM_maxdd = calcMaximumDrawdown(splicedMarketDF)
# print("max DD")
# print(BM_maxdd)
# print(" ")

# BM_volatility = calcVolatility(splicedMarketDF)
# print("volatility")
# print(BM_volatility)
# print(" ")

# BM_yearly_return = calcYearlyReturn(splicedMarketDF, numYears)
# print("Yearly return")
# print(BM_yearly_return)
# print(" ")

# alphaDifference = yearlyReturn - BM_yearly_return

# print("ALPHA DIFFERENCE")
# print(alphaDifference)


# graphPriceData(combined_min_var_dataframe, marketDFPriceDF)


