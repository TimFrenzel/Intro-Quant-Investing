import csv
import pandas as pd
from datetime import datetime
import numpy as np
from itertools import islice

def parse_csv_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the headers from the first row
        date_header = headers[0]  # Assuming the first column contains dates
        for header in headers[1:]:  # Start from the second column for headers
            data[header] = {}

        prev_prices = {}  # Store previous prices for each index
        for row in reader:
            date = row[0]  # Assuming the date is in the first column
            for i, header in enumerate(headers[1:]):  # Start from the second column for data
                if row[i+1]:  # Check if the value in the cell is not empty
                    value = row[i+1]
                    try:
                        float_value = float(value)
                        
                        data[header][date] = float_value  # Assuming values are numerical, convert to float
                        prev_prices[header] = float_value  # Update previous price
                    except ValueError:
                        data[header][date] = np.nan
                        prev_prices[header] = np.nan
                else:  # If the value is empty, use previous day's price
                    data[header][date] = prev_prices.get(header, None)

    return data

import pandas as pd
import numpy as np

def parse_csv_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        headers = file.readline().strip().split(",")  # Read the headers from the first line
        headers = [header.strip() for header in headers[1:]]  # Remove any leading or trailing whitespaces

        for line in file:
            row = line.strip().split(",")  # Split the line into columns
            date = row[0]  # Assuming the date is in the first column
            prices = []
            for val in row[1:]:
                try:
                    prices.append(float(val))
                except ValueError:
                    prices.append(np.nan)  # Replace non-numeric values with NaN
            data.append([date] + prices)
    #print(data)
    pricesDF = pd.DataFrame(data, columns=['Dates'] + headers)
    
    if pricesDF.iloc[:, -1].isnull().all():
        pricesDF = pricesDF.iloc[:, :-1]  # Drop the last column if it consists entirely of NaN values
    
    
    return pricesDF



# def process_input_file(input_file, output_file, largest_V, numRows,v_Intervals=None):
#     # Function to extract values from a specific column starting from a certain row
#     def extract_value(filename, row, column):
#         value = None
#         with open(filename, "r") as f:
#             # Skip rows until the start row
#             for _ in range(row - 1):
#                 next(f)
#             # Read and process the specified row
#             for idx, line in enumerate(f, row):
#                 # Split the line by comma
#                 row_values = line.strip().split(",")
#                 # Check if the column index is valid
#                 if 0 < column <= len(row_values):
#                     value = row_values[column - 1]
#                 else:
#                     # Handle the case where the column index is out of range
#                     print(f"Column index {column} is out of range for row {row}")
#                 # Break out of the loop after extracting the value
#                 break
#         return value

#     # Generate array of V values based on intervals
#     v_array = [1,2]  # Initialize with V2
#     for j in range(1, largest_V):
#         if j % v_Intervals == 0:
#             x = j + 2  # Adjust index to match column numbers (1-based indexing)
#             v_array.append(x)

#     # Write the extracted values to the output file
#     with open(output_file, "w") as f:
#         # Write the header
#         # f.write(",".join([f"V{v}" for v in v_array]) + "\n")

#         # Extract and write values for each V column
#         for i in range(1,numRows):
#                 #row_values = []
#             for v in v_array:
#                 if i == 1:
#                     z=0
#                     pass
#                 elif v ==1 and i ==2:
#                     f.write("Dates,")
#                     z=0
                    
#                 elif i ==3 or i ==4:
#                     #f.write("Dates,")
#                     z=0
#                     pass
#                 elif i ==5:
#                     z=0
#                     pass
#                 else:
#                     z =1
#                     value = extract_value(input_file, i, column=v)
#                     # row_values.append(value)
#         # Write the values row by row
#                     #print(value)
#                     f.write(value + ",")
#             if(z==1):
#                 f.write("\n")


def process_input_file(input_file, output_file, largest_V, numRows, v_Intervals=None):
    def read_file_into_dataframe(input_file):
        df = pd.read_csv(input_file)
        return df
    
    def extract_value(inputDF, row_value, column):
        value = inputDF.iloc[row_value -1, column -1]
        return value


    v_array = [1, 2]  # Initialize with V2
    for j in range(1, largest_V):
        if j % v_Intervals == 0:
            x = j + 2
            v_array.append(x)
    inputDF = read_file_into_dataframe(input_file)
    print(inputDF)
    with open(output_file, "w") as f:
        # Write the header
        # f.write(",".join([f"V{v}" for v in v_array]) + "\n")

        # Extract and write values for each V column
        for i in range(0,numRows):
                #row_values = []
            for v in v_array:
                if i ==0:
                    value = extract_value(inputDF, i, column=v)
                    z = 1
                if i == 1 or i ==2:
                    z=0
                    pass
                # elif v ==1 and i ==3:
                #     f.write("Dates,")
                #     z=0
                elif i ==3 and v!= 1:
                    z=0
                    pass
                else:
                    z =1
                    
                    value = extract_value(inputDF, i, column=v)
                    # row_values.append(value)
        # Write the values row by row
                    #print(value)
                    f.write(str(value) + ",")

            if(z==1):
                f.write("\n")

def normalizePrice(pricesDF):
    dailyReturnsDF = pricesToDailyReturns(pricesDF, ["SPX Index"])
    
    dailyreturnslist = dailyReturnsDF["SPX Index"].tolist()
    dates = pricesDF["Dates"].tolist()
    pricesList = [100]
    for value in dailyreturnslist:
        price = pricesList[-1] * value
        pricesList.append(price)
    

    dataDict = {"Dates": dates, "SPX Index": pricesList}
    df = pd.DataFrame(dataDict)
    
    return df


def createDailyReturnDF(pricesDF, stockList, dataOrder, startDate, endDate): #this is just to grab the price data, the stocks are already data frames

    dates = pricesDF.iloc[:, 0].tolist()

    dateObjects = convert_to_datetime(dates)
    
    #dates = [datetime.strptime(date_str, dateFormat) for date_str in dates]  # Convert strings to datetime objects


    stockReturnsValues = []
    
    if(dataOrder == 'd'):
        dateObjects.reverse()
    
    for stock in stockList:
        prices = list(pricesDF[stock])

        if(dataOrder == 'd'):
            prices.reverse()#DELETE THE REVERSE SECTION WHEN I USE A PROPERLY ORDERED FILE

        prices_list = []
# Initialize an empty list to store the corresponding cumulative returns
        for price in prices:
            if isinstance(price, (int, float)) and not np.isnan(price):
                new_price = ((price/100) +1) * 100
            else:
                new_price = price
            prices_list.append(new_price)
        stockReturnsValues.append(prices_list)

    data_dict = {'Dates': dateObjects}

    for i, stock_name in enumerate(stockList):
        data_dict[stock_name] = stockReturnsValues[i]

    stockPricesDF = pd.DataFrame(data_dict)
    #print(stockPricesDF)

    index_of_startDate = stockPricesDF.loc[stockPricesDF['Dates'] == startDate].index[0]
    index_of_endDate = stockPricesDF.loc[stockPricesDF['Dates'] == endDate].index[0]

    # print(index_of_startDate)
    # print(index_of_endDate)
    
    indexedDF = stockPricesDF[index_of_startDate: index_of_endDate +1]
    indexedDF.reset_index(drop=True, inplace=True)
    #print(indexedDF)
    #indexedDF is just cumulative return data from the raw price data. 
    #it is also indexed to have only the correct date values
    dailyReturnDF = pricesToDailyReturns(indexedDF, stockList) 
    
    return dailyReturnDF, dateObjects




def pricesToDailyReturns(stock_df, stockList): #
    # Extract prices from the DataFrame
    
    stockDailyReturnsList = []

    dates = stock_df["Dates"].tolist()
    del dates[0]
    
    for stock in stockList:
        
        price_values = stock_df[stock].values

        returns = []

        for i in range(1, len(price_values)):
            if isinstance(price_values[i-1], (int, float)) and not np.isnan(price_values[i-1]) and isinstance(price_values[i], (int, float)): #if current and previous price are both numbers
                daily_return = (price_values[i] - price_values[i-1]) / price_values[i-1]  # calculate daily return

                daily_return += 1
                daily_return = np.around(daily_return, 6)
                returns.append(daily_return)

            elif isinstance(price_values[i-1], (int, float)) and not np.isnan(price_values[i-1] and np.isnan(price_values[i])): #if previous price is a number, but current price is nan, return a 1
                daily_return = 1 #this handles the case where the previous number is nan, but the next number is not nan
                returns.append(daily_return)
            
            elif isinstance(price_values[i], (int, float)) and not np.isnan(price_values[i]) and np.isnan(price_values[i -1]) and isinstance(price_values[i-2], (int, float)): #if previous price is a nan, but current price is number, and 2 day before is number, 
                daily_return = (price_values[i] - price_values[i-2]) / price_values[i-2]
                returns.append(daily_return)
            else:
                returns.append(price_values[i])

        stockDailyReturnsList.append(returns)
    #print(stockDailyReturnsList)
    data_dict = {'Dates': dates}


    for i, stock_name in enumerate(stockList):
        data_dict[stock_name] = stockDailyReturnsList[i]
    #print(len(stockDailyReturnsList[0]))
    #print(len(dates))

    #print(stock_df)
    z = stock_df.drop(index = 0)
    #print(z)

    DailyReturnsDF = pd.DataFrame(data_dict, index=z.index)

    return DailyReturnsDF




def calcBeta(stockDailyReturns, marketDailyReturns):
    #print(stockDailyReturns)
    covariance = np.cov(stockDailyReturns, marketDailyReturns)[0, 1]
    market_variance = np.var(marketDailyReturns)
    beta = covariance / market_variance
    #print(beta)
    return beta

def calculate_timeframe_returns(daily_returns, window_size):
    num_days = len(daily_returns)
    num_windows = num_days // window_size
    monthly_returns = []

    for i in range(num_windows):
        start_index = i * window_size
        end_index = min(start_index + window_size, num_days)
        window_returns = daily_returns[start_index:end_index]
        monthly_returns.append(sum(window_returns))

    return monthly_returns

# def makeDailyBetaDfs(stocksDailyReturnsDFs, marketDailyReturnsDFs, stockList, dates):
#     # print(marketDailyReturnsDFs.keys())
#     marketDailyReturns = marketDailyReturnsDFs["SPX Index"].tolist()
    
#     stockBetasList = []
#     #print(len(stockList))

#     for stock in stockList[1:]:
#         stockBetaValues = []
#         stockDailyReturns = stocksDailyReturnsDFs[stock].tolist()

#         for date in dates:
#             index_of_date = stocksDailyReturnsDFs.loc[stocksDailyReturnsDFs['Dates'] == date].index[0]
#             first_row_index = stocksDailyReturnsDFs.index[0]
#             if index_of_date - 756 > first_row_index: #originally was 126
                
#                 index_of_section_start = index_of_date - 756 #was 126
            
#                 stockDailyReturnsSection = stockDailyReturns[index_of_section_start:index_of_date +1]
#                 marketDailyReturnsSection = marketDailyReturns[index_of_section_start:index_of_date +1]
#                 stockTimeframeReturn = calculate_timeframe_returns(stockDailyReturnsSection, 28)
#                 marketTimeFrameReturn = calculate_timeframe_returns(marketDailyReturnsSection, 28)
#                 stockBeta = calcBeta(stockTimeframeReturn, marketTimeFrameReturn)
#                 stockBetaValues.append(stockBeta)
#         #print(stockBetaValues)
#         stockBetasList.append(stockBetaValues)
        
#     betaDates = dates[757:] #was 127
#     betaDatesObjects = convert_to_datetime(betaDates)

#     data_dict = {'Dates': betaDatesObjects}
    
#     for i, stock_name in enumerate(stockList[1:], start=1):
#         data_dict[stock_name] = stockBetasList[i-1]
        
#     # print(len(betaDates))
#     # print(len(data_dict["AGILENT TECHNOLOGIES INC"]))
        
#     stockBetasDf = pd.DataFrame(data_dict)
#     betaDatesObjects = convert_to_datetime(betaDates)
    
#     return stockBetasDf, betaDatesObjects

# def makeDailyBetaDfs(stocksDailyReturnsDFs, marketDailyReturnsDFs, stockList, dates, numBetaLookbackDays, numBetaGroupingDays):
#     # print(marketDailyReturnsDFs.keys())
#     marketDailyReturns = marketDailyReturnsDFs["SPX Index"].tolist()
    
#     stockBetasList = []
#     #print(len(stockList))

#     for stock in stockList[1:]:
#         stockBetaValues = []
#         stockDailyReturns = stocksDailyReturnsDFs[stock].tolist()

#         for date in dates:
#             index_of_date = stocksDailyReturnsDFs.loc[stocksDailyReturnsDFs['Dates'] == date].index[0]
#             first_row_index = stocksDailyReturnsDFs.index[0]
#             if index_of_date - numBetaLookbackDays > first_row_index: #originally was 126
                
#                 index_of_section_start = index_of_date - numBetaLookbackDays #was 126
            
#                 stockDailyReturnsSection = stockDailyReturns[index_of_section_start:index_of_date +1]
#                 marketDailyReturnsSection = marketDailyReturns[index_of_section_start:index_of_date +1]
#                 stockTimeframeReturn = calculate_timeframe_returns(stockDailyReturnsSection, numBetaGroupingDays)
#                 marketTimeFrameReturn = calculate_timeframe_returns(marketDailyReturnsSection, numBetaGroupingDays)
#                 stockBeta = calcBeta(stockTimeframeReturn, marketTimeFrameReturn)
#                 stockBetaValues.append(stockBeta)
#         #print(stockBetaValues)
#         stockBetasList.append(stockBetaValues)
        
#     betaDates = dates[numBetaLookbackDays + 1:] #was 127
#     betaDatesObjects = convert_to_datetime(betaDates)

#     data_dict = {'Dates': betaDatesObjects}
    
#     for i, stock_name in enumerate(stockList[1:], start=1):
#         data_dict[stock_name] = stockBetasList[i-1]
        
#     # print(len(betaDates))
#     # print(len(data_dict["AGILENT TECHNOLOGIES INC"]))
        
#     stockBetasDf = pd.DataFrame(data_dict)
#     # betaDatesObjects = convert_to_datetime(betaDates)
    
#     return stockBetasDf, betaDatesObjects

def makeDailyBetaDfs(stocksDailyReturnsDFs, marketDailyReturnsDFs, stockList, dates, numBetaLookbackDays, numBetaGroupingDays):
    # print(marketDailyReturnsDFs.keys())
    marketDailyReturns = marketDailyReturnsDFs["SPX Index"].tolist()
    
    stockBetasList = []
    #print(len(stockList))

    for stock in stockList[1:]:
        stockBetaValues = []
        stockDailyReturns = stocksDailyReturnsDFs[stock].tolist()

        for date in dates:
            index_of_date = stocksDailyReturnsDFs.loc[stocksDailyReturnsDFs['Dates'] == date].index[0]
            first_row_index = stocksDailyReturnsDFs.index[0]
            if index_of_date - numBetaLookbackDays > first_row_index: #originally was 126
                
                index_of_section_start = index_of_date - numBetaLookbackDays #was 126
            
                stockDailyReturnsSection = stockDailyReturns[index_of_section_start:index_of_date +1]
                marketDailyReturnsSection = marketDailyReturns[index_of_section_start:index_of_date +1]
                stockTimeframeReturn = calculate_timeframe_returns(stockDailyReturnsSection, numBetaGroupingDays)
                marketTimeFrameReturn = calculate_timeframe_returns(marketDailyReturnsSection, numBetaGroupingDays)
                stockBeta = calcBeta(stockTimeframeReturn, marketTimeFrameReturn)
                stockBetaValues.append(stockBeta)
        #print(stockBetaValues)
        stockBetasList.append(stockBetaValues)
        
    betaDates = dates[numBetaLookbackDays + 1:] #was 127
    betaDatesObjects = convert_to_datetime(betaDates)

    data_dict = {'Dates': betaDatesObjects}
    
    for i, stock_name in enumerate(stockList[1:], start=1):
        data_dict[stock_name] = stockBetasList[i-1]
        
    # print(len(betaDates))
    # print(len(data_dict["AGILENT TECHNOLOGIES INC"]))
        
    stockBetasDf = pd.DataFrame(data_dict)
    # betaDatesObjects = convert_to_datetime(betaDates)
    
    return stockBetasDf

    
def remove_additional_commas(input_file, output_file):
    # Read the input file and remove additional comma columns
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove additional comma columns
        cleaned_line = ",".join([entry for entry in line.strip().split(",") if entry.strip()])
        cleaned_lines.append(cleaned_line)

    # Write the modified content into the output file
    with open(output_file, "w") as outfile:
        for line in cleaned_lines:
            outfile.write(line + "\n")
            
def getQuarterDates(dates):
    quarterChangesDates = []
    
    for i in range(len(dates)):
        if i+1 != len(dates):
            currentDateQ = getQuarter(dates[i])
            nextDateQ = getQuarter(dates[i+1])
        else:
            currentDateQ = getQuarter(dates[i])
            nextDateQ = getQuarter(dates[i])
        if(currentDateQ != nextDateQ):
            quarterChangesDates.append(dates[i])
    
    return quarterChangesDates

def getQuarter(date):
    month = date.month
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    else:
        return 4
    
    
    
def determine_date_format(date_string):
    possible_formats = [
        '%Y-%m-%d',  # YYYY-MM-DD
        '%m-%d-%Y',  # MM-DD-YYYY
        '%d-%m-%Y',  # DD-MM-YYYY
        '%m/%d/%y',
        '%m/%d/%Y', 
        # Add more possible formats as needed
    ]

    for date_format in possible_formats:
        try:
            datetime.strptime(date_string, date_format)
            return date_format
        except ValueError:
            pass

    # If none of the formats match, return None
    return None

            
def convert_to_datetime(date_strings):
    date_objects = []
    for date_str in date_strings:
        date_object = datetime.strptime(date_str, "%m/%d/%Y")  # Adjust format string as per your date format
        date_objects.append(date_object)
    return date_objects


def convert_to_datetime2(date_strings):
    date_objects = []
    for date_str in date_strings:
        date_object = datetime.strptime(date_str, '%Y-%m-%d')  # Adjust format string as per your date format
        date_objects.append(date_object)
    return date_objects
        
def convert_to_datetime3(date_strings):
    date_objects = []
    for date_str in date_strings:
        date_object = datetime.strptime(date_str, "%m/%d/%Y")  # Adjust format string as per your date format
        date_objects.append(date_object)
    return date_objects
        
        
        