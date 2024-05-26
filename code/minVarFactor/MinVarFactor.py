from fetchDataAndMakeDFs import parse_csv_file;
from datetime import datetime
import pandas as pd
import numpy as np
from fetchDataAndMakeDFs import convert_to_datetime

# def getLongAndShortBetaPositions(betaQuarterChangeDates, dailyBetasDf):
    
#     absolute_value_betas_df = dailyBetasDf.apply(absolute_value_except_first, axis=1)
    
#     list_of_high_beta_names = []
#     list_of_low_beta_names = []


#     print(betaQuarterChangeDates)
#     print(dailyBetasDf)
#     # print(absolute_value_betas_df)
    
#     for date in betaQuarterChangeDates:
#         print(date)
#         # print(betaQuarterChangeDates)
#         index_of_date = dailyBetasDf.loc[dailyBetasDf['Dates'] == date].index[0]
#         # print(date)
#         # print(index_of_date)
#         # print(dailyBetasDf.loc[dailyBetasDf['Dates'] == date])
#         # row = dailyBetasDf.loc[dailyBetasDf['Dates'] == date]
#         row = absolute_value_betas_df.iloc[index_of_date]
#         # row = dailyBetasDf.iloc[index_of_date]


        
#         row_values = row.iloc[1:]
        
#         # row_values = row_values.values.flatten()

        
#         # row_values = pd.Series(row_values)


#         row_values_numeric = pd.to_numeric(row_values, errors='coerce')


#         highest_betas = row_values_numeric.nlargest(10)
#         highest_betas_names = highest_betas.index.tolist()
#         list_of_high_beta_names.append(highest_betas_names)

#         lowest_betas = row_values_numeric.nsmallest(10)
#         lowest_betas_names = lowest_betas.index.tolist()
#         list_of_low_beta_names.append(lowest_betas_names)

#     return list_of_high_beta_names, list_of_low_beta_names
    

# def minVarFactorPorfolio(list_of_high_beta_names, list_of_low_beta_names, betaQuarterChangeDates, dailyReturnsDf, stockList, betaDates):

#     min_var_portfolios = []

#     stock_dates_all = dailyReturnsDf["Dates"].tolist()
#     stock_dates_all_objects = convert_to_datetime(stock_dates_all)

    
#     for i in range(len(betaQuarterChangeDates)):
#         quarter_start_date = betaQuarterChangeDates[i]
#         if i == len(betaQuarterChangeDates) -1:
#             quarter_end_date = betaDates[-1]
#         else:
#             # print(i)
#             # print(len(betaQuarterChangeDates))
#             # print("check")
#             quarter_end_date = betaQuarterChangeDates[i+1]
        
#         dates_section_start_index = betaDates.index(quarter_start_date)
#         dates_section_end_index = betaDates.index(quarter_end_date)
#         dates_section_end_index -=1
#         dates_section = betaDates[dates_section_start_index: dates_section_end_index +1] #i got rid of the plus 1 here, so i dont access the last date. Not true <-
#         if i == 0:
#             total_portfolio_value = 100
#             start_quarter_price = total_portfolio_value / 20
#         else:
#             total_portfolio_value = min_var_portfolios[i-1].iloc[-1]["Portfolio"]
#             start_quarter_price = total_portfolio_value / 20

#         all_stock_prices_list = []
        
#         for stock in stockList[1:]:
#             if stock in list_of_low_beta_names[i]: #uses reguar daily return
#                 stock_price_list = [start_quarter_price]
#                 stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#                 daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#                 daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#                 stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] # deleted the +1

#                 investment = 1
            
#             elif stock in list_of_high_beta_names[i]: #uses negative daily return, showing a short position
                
#                 stock_price_list = [start_quarter_price]
#                 stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#                 daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#                 daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#                 # print(len(stock_daily_returns_all))
#                 # print(len(stock_dates_all_objects))
#                 # print(len(betaDates))
#                 # qq
#                 stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] #deleted plus 1 
#                 investment = 2
#                 # if stock == 'BLACKROCK INC':
#                 #     print(quarter_start_date)
#                 #     print(daily_return_start_index)
#                 #     print(daily_return_end_index)
#                 #     ll
#             else:
#                 stock_price_list = [0]
#                 stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#                 #print(stock_dates_all_objects)
#                 daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#                 daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#                 stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index]
#                 investment = 0

#             for daily_return in stock_daily_returns_section:
#                 # if daily_return == 1.005544:
#                 #     print(stock)
#                 #     oo
#                 if investment == 1:
#                     # if np.isnan(daily_return):
#                     #     print("l")
#                     #     print(daily_return)
#                     #     daily_return = 1
#                     #     print(daily_return)
#                     #     print(stock_price_list)
#                     #     stock_price = stock_price_list[-1] * daily_return
#                     #     print(stock_price)
                        
#                     # else:
#                     if np.isnan(stock_price_list[-1]):
#                         print(f'Nan val for long')
#                         print(stock)
#                         # qpwpwp
#                     stock_price = stock_price_list[-1] * daily_return
#                     stock_price_list.append(stock_price)
#                     # if stock_price_list[-1] > 10:
#                     #     print(stock_price_list)
#                     #     print(stock_daily_returns_section)
#                     #     pp
#                 elif investment == 2:
#                     # if np.isnan(daily_return):
#                     #     print("s")
#                     #     print(daily_return)
#                     #     daily_return = 1
#                     #     print(daily_return)
                    
#                     if np.isnan(stock_price_list[-1]):
#                         stock_price_list[-1] = stock_price_list[-2]
#                         print(f'Nan val for short')
#                         print(stock)
                        
#                     short_daily_return = 2 - daily_return 
#                     stock_price = stock_price_list[-1] * short_daily_return
#                     stock_price_list.append(stock_price)
#                     # if stock == 'BLACKROCK INC':
#                     #     print(daily_return)
#                     #     print(short_daily_return)
#                     #     print(stock_price)
#                     #     if i == 1:
#                     #         zz
                        
#                 else:
#                     stock_price_list.append(0)
#                     # if stock == 'BUILDERS FIRSTSOURCE INC':
#                     #     print(daily_return)
#                     #     zz
            
#             stock_price_list.pop(-1) #i used to pop 0

#             all_stock_prices_list.append(stock_price_list)
            
#         min_var_data_dict_section = {"Dates": dates_section}
        

        
#         for i, stock_name in enumerate(stockList[1:], start=1):
#             min_var_data_dict_section[stock_name] = all_stock_prices_list[i-1]
            
#         # print(len(min_var_data_dict_section['Dates']))
#         # print(len(min_var_data_dict_section['BLACKROCK INC']))
#         # print(min_var_data_dict_section['BLACKROCK INC'])
#         # print("foo")
#         # print(dates_section)
#         min_var_data_dict_section_DF = pd.DataFrame(min_var_data_dict_section)

#         z = 0
        
#         portfolio_values = []
        
#         for date in dates_section:
#             portfolio_value_at_date = 0

#             for stock in stockList[1:]:
#                 date_index = min_var_data_dict_section_DF[stock].loc[min_var_data_dict_section_DF["Dates"] == date].index[0]
#                 stock_value_at_date = min_var_data_dict_section_DF.loc[date_index, stock]
#                 # if np.isnan(stock_value_at_date):
#                 #     listofvals = min_var_data_dict_section_DF[stock].tolist()
#                 #     print(listofvals)
#                 #     print(stock_value_at_date)
#                 #     print(date)
#                 #     print("check")
#                 #     print(z)
#                 #     print(stock)
#                 #     ad
#                 #     z+=1
#                 portfolio_value_at_date += stock_value_at_date
#             #     if np.isnan(portfolio_value_at_date):
#             #         print(stock)
#             #         qq
#             # if np.isnan(portfolio_value_at_date):
#             #     print(date)
#             #     aa
                    
#             portfolio_values.append(portfolio_value_at_date)
        
#         min_var_data_dict_section_DF.insert(1, "Portfolio", portfolio_values)
            
#         min_var_portfolios.append(min_var_data_dict_section_DF)
        
        

        
#     return min_var_portfolios



def getLongAndShortBetaPositions(betaQuarterChangeDates, dailyBetasDf):
    
    absolute_value_betas_df = dailyBetasDf.apply(absolute_value_except_first, axis=1)
    
    list_of_high_beta_names = []
    list_of_low_beta_names = []


    print(betaQuarterChangeDates)
    print(dailyBetasDf)
    # print(absolute_value_betas_df)
    
    for date in betaQuarterChangeDates:
        print(date)
        # print(betaQuarterChangeDates)
        index_of_date = dailyBetasDf.loc[dailyBetasDf['Dates'] == date].index[0]
        # print(date)
        # print(index_of_date)
        # print(dailyBetasDf.loc[dailyBetasDf['Dates'] == date])
        # row = dailyBetasDf.loc[dailyBetasDf['Dates'] == date]
        row = absolute_value_betas_df.iloc[index_of_date]
        # row = dailyBetasDf.iloc[index_of_date]


        
        row_values = row.iloc[1:]
        
        # row_values = row_values.values.flatten()

        
        # row_values = pd.Series(row_values)


        row_values_numeric = pd.to_numeric(row_values, errors='coerce')


        highest_betas = row_values_numeric.nlargest(10)
        highest_betas_names = highest_betas.index.tolist()
        list_of_high_beta_names.append(highest_betas_names)

        lowest_betas = row_values_numeric.nsmallest(10)
        lowest_betas_names = lowest_betas.index.tolist()
        list_of_low_beta_names.append(lowest_betas_names)

    return list_of_high_beta_names, list_of_low_beta_names
    

def minVarFactorPorfolio(list_of_high_beta_names, list_of_low_beta_names, betaQuarterChangeDates, dailyReturnsDf, stockList, betaDates):

    min_var_portfolios = []

    stock_dates_all = dailyReturnsDf["Dates"].tolist()
    stock_dates_all_objects = convert_to_datetime(stock_dates_all)

    
    for i in range(len(betaQuarterChangeDates)):
        quarter_start_date = betaQuarterChangeDates[i]
        if i == len(betaQuarterChangeDates) -1:
            quarter_end_date = betaDates[-1]
        else:
            # print(i)
            # print(len(betaQuarterChangeDates))
            # print("check")
            quarter_end_date = betaQuarterChangeDates[i+1]
        
        dates_section_start_index = betaDates.index(quarter_start_date)
        dates_section_end_index = betaDates.index(quarter_end_date)
        dates_section_end_index -=1
        dates_section = betaDates[dates_section_start_index: dates_section_end_index +1] #i got rid of the plus 1 here, so i dont access the last date. Not true <-
        if i == 0:
            total_portfolio_value = 100
            start_quarter_price = total_portfolio_value / 20
        else:
            total_portfolio_value = min_var_portfolios[i-1].iloc[-1]["Portfolio"]
            start_quarter_price = total_portfolio_value / 20

        all_stock_prices_list = []
        
        for stock in stockList[1:]:
            if stock in list_of_low_beta_names[i]: #uses reguar daily return
                stock_price_list = [start_quarter_price]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] # deleted the +1

                investment = 1
            
            elif stock in list_of_high_beta_names[i]: #uses negative daily return, showing a short position
                
                stock_price_list = [start_quarter_price]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                # print(len(stock_daily_returns_all))
                # print(len(stock_dates_all_objects))
                # print(len(betaDates))
                # qq
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] #deleted plus 1 
                investment = 2
                # if stock == 'BLACKROCK INC':
                #     print(quarter_start_date)
                #     print(daily_return_start_index)
                #     print(daily_return_end_index)
                #     ll
            else:
                stock_price_list = [0]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                #print(stock_dates_all_objects)
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index]
                investment = 0

            for daily_return in stock_daily_returns_section:
                # if daily_return == 1.005544:
                #     print(stock)
                #     oo
                if investment == 1:
                    # if np.isnan(daily_return):
                    #     print("l")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    #     print(stock_price_list)
                    #     stock_price = stock_price_list[-1] * daily_return
                    #     print(stock_price)
                        
                    # else:
                    if np.isnan(stock_price_list[-1]):
                        print(f'Nan val for long')
                        print(stock)
                        # qpwpwp
                    stock_price = stock_price_list[-1] * daily_return
                    stock_price_list.append(stock_price)
                    # if stock_price_list[-1] > 10:
                    #     print(stock_price_list)
                    #     print(stock_daily_returns_section)
                    #     pp
                elif investment == 2:
                    # if np.isnan(daily_return):
                    #     print("s")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    
                    if np.isnan(stock_price_list[-1]):
                        stock_price_list[-1] = stock_price_list[-2]
                        print(f'Nan val for short')
                        print(stock)
                        
                    short_daily_return = 2 - daily_return 
                    stock_price = stock_price_list[-1] * short_daily_return
                    stock_price_list.append(stock_price)
                    # if stock == 'BLACKROCK INC':
                    #     print(daily_return)
                    #     print(short_daily_return)
                    #     print(stock_price)
                    #     if i == 1:
                    #         zz
                        
                else:
                    stock_price_list.append(0)
                    # if stock == 'BUILDERS FIRSTSOURCE INC':
                    #     print(daily_return)
                    #     zz
            
            stock_price_list.pop(-1) #i used to pop 0

            all_stock_prices_list.append(stock_price_list)
            
        min_var_data_dict_section = {"Dates": dates_section}
        

        
        for i, stock_name in enumerate(stockList[1:], start=1):
            min_var_data_dict_section[stock_name] = all_stock_prices_list[i-1]
            
        # print(len(min_var_data_dict_section['Dates']))
        # print(len(min_var_data_dict_section['BLACKROCK INC']))
        # print(min_var_data_dict_section['BLACKROCK INC'])
        # print("foo")
        # print(dates_section)
        min_var_data_dict_section_DF = pd.DataFrame(min_var_data_dict_section)

        z = 0
        
        portfolio_values = []
        
        for date in dates_section:
            portfolio_value_at_date = 0

            for stock in stockList[1:]:
                date_index = min_var_data_dict_section_DF[stock].loc[min_var_data_dict_section_DF["Dates"] == date].index[0]
                stock_value_at_date = min_var_data_dict_section_DF.loc[date_index, stock]
                # if np.isnan(stock_value_at_date):
                #     listofvals = min_var_data_dict_section_DF[stock].tolist()
                #     print(listofvals)
                #     print(stock_value_at_date)
                #     print(date)
                #     print("check")
                #     print(z)
                #     print(stock)
                #     ad
                #     z+=1
                portfolio_value_at_date += stock_value_at_date
            #     if np.isnan(portfolio_value_at_date):
            #         print(stock)
            #         qq
            # if np.isnan(portfolio_value_at_date):
            #     print(date)
            #     aa
                    
            portfolio_values.append(portfolio_value_at_date)
        
        min_var_data_dict_section_DF.insert(1, "Portfolio", portfolio_values)
            
        min_var_portfolios.append(min_var_data_dict_section_DF)
        
        

        
    return min_var_portfolios



def minVarFactorPorfolioLong(list_of_high_beta_names, list_of_low_beta_names, betaQuarterChangeDates, dailyReturnsDf, stockList, betaDates):

    min_var_portfolios = []

    stock_dates_all = dailyReturnsDf["Dates"].tolist()
    stock_dates_all_objects = convert_to_datetime(stock_dates_all)

    
    for i in range(len(betaQuarterChangeDates)):
        quarter_start_date = betaQuarterChangeDates[i]
        if i == len(betaQuarterChangeDates) -1:
            quarter_end_date = betaDates[-1]
        else:
            # print(i)
            # print(len(betaQuarterChangeDates))
            # print("check")
            quarter_end_date = betaQuarterChangeDates[i+1]
        
        dates_section_start_index = betaDates.index(quarter_start_date)
        dates_section_end_index = betaDates.index(quarter_end_date)
        dates_section_end_index -=1
        dates_section = betaDates[dates_section_start_index: dates_section_end_index +1] #i got rid of the plus 1 here, so i dont access the last date. Not true <-
        if i == 0:
            total_portfolio_value = 100
            start_quarter_price = total_portfolio_value / 10
        else:
            total_portfolio_value = min_var_portfolios[i-1].iloc[-1]["Portfolio"]
            start_quarter_price = total_portfolio_value / 10

        all_stock_prices_list = []
        
        for stock in stockList[1:]:
            if stock in list_of_low_beta_names[i]: #uses reguar daily return
                stock_price_list = [start_quarter_price]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] # deleted the +1

                investment = 1
            
            # elif stock in list_of_high_beta_names[i]: #uses negative daily return, showing a short position
                
            #     # stock_price_list = [start_quarter_price]
            #     # stock_daily_returns_all = dailyReturnsDf[stock].tolist()
            #     # daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
            #     # daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
            #     # # print(len(stock_daily_returns_all))
            #     # # print(len(stock_dates_all_objects))
            #     # # print(len(betaDates))
            #     # # qq
            #     # stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] #deleted plus 1 
            #     investment = 0
                # if stock == 'BLACKROCK INC':
                #     print(quarter_start_date)
                #     print(daily_return_start_index)
                #     print(daily_return_end_index)
                #     ll
            else:
                stock_price_list = [0]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                #print(stock_dates_all_objects)
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index]
                investment = 0

            for daily_return in stock_daily_returns_section:
                # if daily_return == 1.005544:
                #     print(stock)
                #     oo
                if investment == 1:
                    # if np.isnan(daily_return):
                    #     print("l")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    #     print(stock_price_list)
                    #     stock_price = stock_price_list[-1] * daily_return
                    #     print(stock_price)
                        
                    # else:
                    if np.isnan(stock_price_list[-1]):
                        print(f'Nan val for long')
                        print(stock)
                        # qpwpwp
                    stock_price = stock_price_list[-1] * daily_return
                    stock_price_list.append(stock_price)
                    # if stock_price_list[-1] > 10:
                    #     print(stock_price_list)
                    #     print(stock_daily_returns_section)
                    #     pp
                elif investment == 2:
                    # if np.isnan(daily_return):
                    #     print("s")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    
                    if np.isnan(stock_price_list[-1]):
                        stock_price_list[-1] = stock_price_list[-2]
                        print(f'Nan val for short')
                        print(stock)
                        
                    short_daily_return = 2 - daily_return 
                    stock_price = stock_price_list[-1] * short_daily_return
                    stock_price_list.append(stock_price)
                    # if stock == 'BLACKROCK INC':
                    #     print(daily_return)
                    #     print(short_daily_return)
                    #     print(stock_price)
                    #     if i == 1:
                    #         zz
                        
                else:
                    stock_price_list.append(0)
                    # if stock == 'BUILDERS FIRSTSOURCE INC':
                    #     print(daily_return)
                    #     zz
            
            stock_price_list.pop(-1) #i used to pop 0

            all_stock_prices_list.append(stock_price_list)
            
        min_var_data_dict_section = {"Dates": dates_section}
        

        
        for i, stock_name in enumerate(stockList[1:], start=1):
            min_var_data_dict_section[stock_name] = all_stock_prices_list[i-1]
            
        # print(len(min_var_data_dict_section['Dates']))
        # print(len(min_var_data_dict_section['BLACKROCK INC']))
        # print(min_var_data_dict_section['BLACKROCK INC'])
        # print("foo")
        # print(dates_section)
        min_var_data_dict_section_DF = pd.DataFrame(min_var_data_dict_section)

        z = 0
        
        portfolio_values = []
        
        for date in dates_section:
            portfolio_value_at_date = 0

            for stock in stockList[1:]:
                date_index = min_var_data_dict_section_DF[stock].loc[min_var_data_dict_section_DF["Dates"] == date].index[0]
                stock_value_at_date = min_var_data_dict_section_DF.loc[date_index, stock]
                # if np.isnan(stock_value_at_date):
                #     listofvals = min_var_data_dict_section_DF[stock].tolist()
                #     print(listofvals)
                #     print(stock_value_at_date)
                #     print(date)
                #     print("check")
                #     print(z)
                #     print(stock)
                #     ad
                #     z+=1
                portfolio_value_at_date += stock_value_at_date
            #     if np.isnan(portfolio_value_at_date):
            #         print(stock)
            #         qq
            # if np.isnan(portfolio_value_at_date):
            #     print(date)
            #     aa
                    
            portfolio_values.append(portfolio_value_at_date)
        
        min_var_data_dict_section_DF.insert(1, "Portfolio", portfolio_values)
            
        min_var_portfolios.append(min_var_data_dict_section_DF)
        
        

        
    return min_var_portfolios



def minVarFactorPorfolioShort(list_of_high_beta_names, list_of_low_beta_names, betaQuarterChangeDates, dailyReturnsDf, stockList, betaDates):

    min_var_portfolios = []

    stock_dates_all = dailyReturnsDf["Dates"].tolist()
    stock_dates_all_objects = convert_to_datetime(stock_dates_all)

    
    for i in range(len(betaQuarterChangeDates)):
        quarter_start_date = betaQuarterChangeDates[i]
        if i == len(betaQuarterChangeDates) -1:
            quarter_end_date = betaDates[-1]
        else:
            # print(i)
            # print(len(betaQuarterChangeDates))
            # print("check")
            quarter_end_date = betaQuarterChangeDates[i+1]
        
        dates_section_start_index = betaDates.index(quarter_start_date)
        dates_section_end_index = betaDates.index(quarter_end_date)
        dates_section_end_index -=1
        dates_section = betaDates[dates_section_start_index: dates_section_end_index +1] #i got rid of the plus 1 here, so i dont access the last date. Not true <-
        if i == 0:
            total_portfolio_value = 100
            start_quarter_price = total_portfolio_value / 10
        else:
            total_portfolio_value = min_var_portfolios[i-1].iloc[-1]["Portfolio"]
            start_quarter_price = total_portfolio_value / 10

        all_stock_prices_list = []
        
        for stock in stockList[1:]:
            # if stock in list_of_low_beta_names[i]: #uses reguar daily return
            #     stock_price_list = [start_quarter_price]
            #     stock_daily_returns_all = dailyReturnsDf[stock].tolist()
            #     daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
            #     daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
            #     stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] # deleted the +1

            #     investment = 1
            
            if stock in list_of_high_beta_names[i]: #uses negative daily return, showing a short position
                
                stock_price_list = [start_quarter_price]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                # print(len(stock_daily_returns_all))
                # print(len(stock_dates_all_objects))
                # print(len(betaDates))
                # qq
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] #deleted plus 1 
                investment = 2
                
                # print(stock_daily_returns_section)
                # if stock == 'BLACKROCK INC':
                #     print(quarter_start_date)
                #     print(daily_return_start_index)
                #     print(daily_return_end_index)
                #     ll
            else:
                stock_price_list = [0]
                stock_daily_returns_all = dailyReturnsDf[stock].tolist()
                #print(stock_dates_all_objects)
                daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
                daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
                stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index]
                investment = 0

            for daily_return in stock_daily_returns_section:
                # if daily_return == 1.005544:
                #     print(stock)
                #     oo
                if investment == 1:
                    # if np.isnan(daily_return):
                    #     print("l")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    #     print(stock_price_list)
                    #     stock_price = stock_price_list[-1] * daily_return
                    #     print(stock_price)
                        
                    # else:
                    if np.isnan(stock_price_list[-1]):
                        print(f'Nan val for long')
                        print(stock)
                        # qpwpwp
                    stock_price = stock_price_list[-1] * daily_return
                    stock_price_list.append(stock_price)
                    # if stock_price_list[-1] > 10:
                    #     print(stock_price_list)
                    #     print(stock_daily_returns_section)
                    #     pp
                elif investment == 2:
                    # if np.isnan(daily_return):
                    #     print("s")
                    #     print(daily_return)
                    #     daily_return = 1
                    #     print(daily_return)
                    
                    if np.isnan(stock_price_list[-1]):
                        stock_price_list[-1] = stock_price_list[-2]
                        print(f'Nan val for short')
                        print(stock)
                        
                    short_daily_return = 2 - daily_return 
                    stock_price = stock_price_list[-1] * short_daily_return
                    stock_price_list.append(stock_price)
                    # if stock == 'BLACKROCK INC':
                    #     print(daily_return)
                    #     print(short_daily_return)
                    #     print(stock_price)
                    #     if i == 1:
                    #         zz
                        
                else:
                    stock_price_list.append(0)
                    # if stock == 'BUILDERS FIRSTSOURCE INC':
                    #     print(daily_return)
                    #     zz
            
            stock_price_list.pop(-1) #i used to pop 0

            all_stock_prices_list.append(stock_price_list)
            
        min_var_data_dict_section = {"Dates": dates_section}
        

        
        for i, stock_name in enumerate(stockList[1:], start=1):
            min_var_data_dict_section[stock_name] = all_stock_prices_list[i-1]
            
        # print(len(min_var_data_dict_section['Dates']))
        # print(len(min_var_data_dict_section['BLACKROCK INC']))
        # print(min_var_data_dict_section['BLACKROCK INC'])
        # print("foo")
        # print(dates_section)
        min_var_data_dict_section_DF = pd.DataFrame(min_var_data_dict_section)

        z = 0
        
        portfolio_values = []
        
        for date in dates_section:
            portfolio_value_at_date = 0

            for stock in stockList[1:]:
                date_index = min_var_data_dict_section_DF[stock].loc[min_var_data_dict_section_DF["Dates"] == date].index[0]
                stock_value_at_date = min_var_data_dict_section_DF.loc[date_index, stock]
                # if np.isnan(stock_value_at_date):
                #     listofvals = min_var_data_dict_section_DF[stock].tolist()
                #     print(listofvals)
                #     print(stock_value_at_date)
                #     print(date)
                #     print("check")
                #     print(z)
                #     print(stock)
                #     ad
                #     z+=1
                portfolio_value_at_date += stock_value_at_date
            #     if np.isnan(portfolio_value_at_date):
            #         print(stock)
            #         qq
            # if np.isnan(portfolio_value_at_date):
            #     print(date)
            #     aa
                    
            portfolio_values.append(portfolio_value_at_date)
        
        min_var_data_dict_section_DF.insert(1, "Portfolio", portfolio_values)
            
        min_var_portfolios.append(min_var_data_dict_section_DF)
        
        

        
    return min_var_portfolios









# def minVarFactorPorfolioShort(list_of_high_beta_names, list_of_low_beta_names, betaQuarterChangeDates, dailyReturnsDf, stockList, betaDates):

#     min_var_portfolios = []

#     stock_dates_all = dailyReturnsDf["Dates"].tolist()
#     stock_dates_all_objects = convert_to_datetime(stock_dates_all)

    
#     for i in range(len(betaQuarterChangeDates)):
#         quarter_start_date = betaQuarterChangeDates[i]
#         if i == len(betaQuarterChangeDates) -1:
#             quarter_end_date = betaDates[-1]
#         else:
#             # print(i)
#             # print(len(betaQuarterChangeDates))
#             # print("check")
#             quarter_end_date = betaQuarterChangeDates[i+1]
        
#         dates_section_start_index = betaDates.index(quarter_start_date)
#         dates_section_end_index = betaDates.index(quarter_end_date)
#         dates_section_end_index -=1
#         dates_section = betaDates[dates_section_start_index: dates_section_end_index +1] #i got rid of the plus 1 here, so i dont access the last date. Not true <-
#         if i == 0:
#             total_portfolio_value = 100
#             start_quarter_price = total_portfolio_value / 10
#         else:
#             total_portfolio_value = min_var_portfolios[i-1].iloc[-1]["Portfolio"]
#             start_quarter_price = total_portfolio_value / 10

#         all_stock_prices_list = []
        
#         for stock in stockList[1:]:
#             # if stock in list_of_low_beta_names[i]: #uses reguar daily return
#             #     stock_price_list = [start_quarter_price]
#             #     stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#             #     daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#             #     daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#             #     stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] # deleted the +1

#             #     investment = 1
            
#             if stock in list_of_high_beta_names[i]: #uses negative daily return, showing a short position
                
#                 stock_price_list = [start_quarter_price]
#                 stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#                 daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#                 daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#                 stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index] #deleted plus 1 
                
#                 investment = 0
#                 # if stock == 'BLACKROCK INC':
#                 #     print(quarter_start_date)
#                 #     print(daily_return_start_index)
#                 #     print(daily_return_end_index)
                    
#             else:
#                 stock_price_list = [0]
#                 stock_daily_returns_all = dailyReturnsDf[stock].tolist()
#                 #print(stock_dates_all_objects)
#                 daily_return_start_index = stock_dates_all_objects.index(quarter_start_date)
#                 daily_return_end_index = stock_dates_all_objects.index(quarter_end_date)
#                 stock_daily_returns_section = stock_daily_returns_all[daily_return_start_index: daily_return_end_index]
#                 investment = 0

#             for daily_return in stock_daily_returns_section:
#                 # if daily_return == 1.005544:
#                 #     print(stock)
#                 #     oo
#                 if investment == 1:
#                     # if np.isnan(daily_return):
#                     #     print("l")
#                     #     print(daily_return)
#                     #     daily_return = 1
#                     #     print(daily_return)
#                     #     print(stock_price_list)
#                     #     stock_price = stock_price_list[-1] * daily_return
#                     #     print(stock_price)
                        
#                     # else:
#                     print('somehow got investment 1')
#                     if np.isnan(stock_price_list[-1]):
#                         print(f'Nan val for long')
#                         print(stock)
#                         # qpwpwp
#                     stock_price = stock_price_list[-1] * daily_return
#                     stock_price_list.append(stock_price)
#                     # if stock_price_list[-1] > 10:
#                     #     print(stock_price_list)
#                     #     print(stock_daily_returns_section)
#                     #     pp
#                 elif investment == 2:
#                     # if np.isnan(daily_return):
#                     #     print("s")
#                     #     print(daily_return)
#                     #     daily_return = 1
#                     #     print(daily_return)
                    
#                     if np.isnan(stock_price_list[-1]):
#                         stock_price_list[-1] = stock_price_list[-2]
#                         print(f'Nan val for short')
#                         print(stock)
                        
#                     short_daily_return = 2 - daily_return 
#                     stock_price = stock_price_list[-1] * short_daily_return
#                     stock_price_list.append(stock_price)
#                     # if stock == 'BLACKROCK INC':
#                     #     print(daily_return)
#                     #     print(short_daily_return)
#                     #     print(stock_price)
#                     #     if i == 1:
#                     #         zz
                        
#                 else:
#                     stock_price_list.append(0)
#                     # if stock == 'BUILDERS FIRSTSOURCE INC':
#                     #     print(daily_return)
#                     #     zz
            
#             stock_price_list.pop(-1) #i used to pop 0

#             all_stock_prices_list.append(stock_price_list)
            
#         min_var_data_dict_section = {"Dates": dates_section}
        

        
#         for i, stock_name in enumerate(stockList[1:], start=1):
#             min_var_data_dict_section[stock_name] = all_stock_prices_list[i-1]
            
#         # print(len(min_var_data_dict_section['Dates']))
#         # print(len(min_var_data_dict_section['BLACKROCK INC']))
#         # print(min_var_data_dict_section['BLACKROCK INC'])
#         # print("foo")
#         # print(dates_section)
#         min_var_data_dict_section_DF = pd.DataFrame(min_var_data_dict_section)

#         z = 0
        
#         portfolio_values = []
        
#         for date in dates_section:
#             portfolio_value_at_date = 0

#             for stock in stockList[1:]:
#                 date_index = min_var_data_dict_section_DF[stock].loc[min_var_data_dict_section_DF["Dates"] == date].index[0]
#                 stock_value_at_date = min_var_data_dict_section_DF.loc[date_index, stock]
#                 # if np.isnan(stock_value_at_date):
#                 #     listofvals = min_var_data_dict_section_DF[stock].tolist()
#                 #     print(listofvals)
#                 #     print(stock_value_at_date)
#                 #     print(date)
#                 #     print("check")
#                 #     print(z)
#                 #     print(stock)
#                 #     ad
#                 #     z+=1
#                 portfolio_value_at_date += stock_value_at_date
#             #     if np.isnan(portfolio_value_at_date):
#             #         print(stock)
#             #         qq
#             # if np.isnan(portfolio_value_at_date):
#             #     print(date)
#             #     aa
                    
#             portfolio_values.append(portfolio_value_at_date)
        
#         min_var_data_dict_section_DF.insert(1, "Portfolio", portfolio_values)
            
#         min_var_portfolios.append(min_var_data_dict_section_DF)
        
        

        
#     return min_var_portfolios






            
    
    
    
    
    
    

    
def multiply_by_negative_one(value):
    if isinstance(value, (int, float)):  # Check if the value is numeric
        return value * -1
    else:
        return value
    
def absolute_value_except_first(row):
    # Initialize an empty list to store modified row values
    modified_row = []
    
    # Iterate over the elements of the row
    for i, val in enumerate(row):
        # Skip the first element (assuming it's not a numeric value)
        if i == 0:
            modified_row.append(val)
        else:
            # Check if the value is numeric and not NaN
            if pd.notnull(val) and np.isfinite(val):
                # Apply the absolute value function to numeric values
                modified_row.append(abs(val))
            else:
                # Keep non-numeric or NaN values unchanged
                modified_row.append(val)
    
    # Convert the modified row list to a pandas Series
    return pd.Series(modified_row, index=row.index)
