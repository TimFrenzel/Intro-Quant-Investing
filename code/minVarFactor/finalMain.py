import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from datetime import datetime as dt
import plotly.graph_objects as go
import pandas as pd
from mainMinVar import runFactorPortfolio3

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.Label('Start Date:'),
    dcc.DatePickerSingle(
        id='date-1-picker',
        date=dt(2009, 12, 31)
    ),
    html.Label('End Date:'),
    dcc.DatePickerSingle(
        id='date-2-picker',
        date=dt(2023, 3, 6)
    ),
    html.Br(),
    html.Label('Beta Lookback Options:'),
    dcc.Dropdown(
        id = 'val1-input',
        options=[
            {'label': '6 Months (126 Days)', 'value': 126},
            {'label': '1 Year (252 Days)', 'value': 252},
            {'label': '3 Years (756 Days)', 'value': 756}
        ],
        value=0
    ),
    html.Label('Beta grouping options:'),
    dcc.Dropdown(
        id = 'val2-input',
        options=[
            {'label': 'Daily', 'value': 1},
            {'label': 'Monthly', 'value': 28}
        ],
        value=0
    ),
    html.Label('Investment Strategy:'),
    dcc.Dropdown(
        id = 'strategy-input',
        options=[
            {'label': 'Contrarian', 'value': 'C'},
            {'label': 'Long', 'value': 'L'},
            {'label': 'Short', 'value': 'S'}
        ],
        value=0
    ),
    html.Button('Run', id='run-button', n_clicks=0),
    dcc.Graph(id='graph-output'),  # Graph component for displaying the graph
    html.Div(id='table-output')  # Div for displaying the table
])

# Define callback to run the function
@app.callback(
    [Output('graph-output', 'figure'), Output('table-output', 'children')],  # Output for graph and table
    [Input('run-button', 'n_clicks')],
    [dash.dependencies.State('date-1-picker', 'date'),
     dash.dependencies.State('date-2-picker', 'date'),
     dash.dependencies.State('val1-input', 'value'),
     dash.dependencies.State('val2-input', 'value'),
     dash.dependencies.State('strategy-input', 'value')]
)
def run_function(n_clicks, date_1, date_2, val1, val2, strategy):
    if n_clicks > 0:
        # Call your function with provided inputs
        outputDF, portfolioXVals, portfolioYVals, marketXVals, marketYVals = runFactorPortfolio3(date_1, date_2, val1, val2, strategy)

        # Prepare data for the graph
        portfolio_df = pd.DataFrame({'Dates': portfolioXVals, 'Portfolio': portfolioYVals})
        market_df = pd.DataFrame({'Dates': marketXVals, 'S&P': marketYVals})

        # Create figure
        fig = go.Figure()

        # Add portfolio line
        fig.add_trace(go.Scatter(x=marketXVals, y=portfolioYVals, mode='lines', name='Portfolio'))

        # Add S&P line
        fig.add_trace(go.Scatter(x=marketXVals, y=marketYVals, mode='lines', name='S&P'))

        fig.update_layout(title='Portfolio vs S&P', xaxis_title='Date', yaxis_title='Value')
        fig.update_yaxes(range=[0, 500])  # Set the bounds of the y-axis

        # Prepare data for the table
        table = dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in outputDF.columns],
            data=outputDF.to_dict('records')
        )

        return fig, table  # Return the figure and table

    else:
        return go.Figure(), html.Div()  # Return empty figure and empty div for the table

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
































# import dash
# from dash import dcc, html, dash_table
# from dash.dependencies import Input, Output
# from datetime import datetime as dt
# import plotly.graph_objects as go
# import pandas as pd
# from mainMinVar import runFactorPortfolio3

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date - After or = 12/31/2009'),
#     dcc.DatePickerSingle(
#         id='date-1-picker',
#         date=dt.today()
#     ),
#     html.Label('End Date - Before or = 3/6/2023'),
#     dcc.DatePickerSingle(
#         id='date-2-picker',
#         date=dt.today()
#     ),
#     html.Label('Beta Lookback Options: 6 Months(Enter 126) - 1 Year(Enter 252) - 3 Year(Enter 756)'),
#     dcc.Input(
#         id='val1-input',
#         type='number',
#         value=0
#     ),
#     html.Label('Beta grouping options: Daily(Enter 1) - Monthly(Enter 28)'),
#     dcc.Input(
#         id='val2-input',
#         type='number',
#         value=0
#     ),
#     html.Button('Run', id='run-button', n_clicks=0),
#     dcc.Graph(id='graph-output'),  # Graph component for displaying the graph
#     html.Div(id='table-output')  # Div for displaying the table
# ])

# # Define callback to run the function
# @app.callback(
#     [Output('graph-output', 'figure'), Output('table-output', 'children')],  # Output for graph and table
#     [Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('date-1-picker', 'date'),
#      dash.dependencies.State('date-2-picker', 'date'),
#      dash.dependencies.State('val1-input', 'value'),
#      dash.dependencies.State('val2-input', 'value')]
# )
# def run_function(n_clicks, date_1, date_2, val1, val2):
#     if n_clicks > 0:
#         # Call your function with provided inputs
#         outputDF, portfolioXVals, portfolioYVals, marketXVals, marketYVals = runFactorPortfolio3(date_1, date_2, val1, val2, 'S')

#         # Prepare data for the graph
#         portfolio_df = pd.DataFrame({'Dates': portfolioXVals, 'Portfolio': portfolioYVals})
#         market_df = pd.DataFrame({'Dates': marketXVals, 'S&P': marketYVals})

#         # Create figure
#         fig = go.Figure()

#         # Add portfolio line
#         fig.add_trace(go.Scatter(x=marketXVals, y=portfolioYVals, mode='lines', name='Portfolio'))

#         # Add S&P line
#         fig.add_trace(go.Scatter(x=marketXVals, y=marketYVals, mode='lines', name='S&P'))

#         fig.update_layout(title='Portfolio vs S&P', xaxis_title='Date', yaxis_title='Value')
#         fig.update_yaxes(range=[0, 500])  # Set the bounds of the y-axis

#         # Prepare data for the table
#         table = dash_table.DataTable(
#             id='table',
#             columns=[{"name": i, "id": i} for i in outputDF.columns],
#             data=outputDF.to_dict('records')
#         )

#         return fig, table  # Return the figure and table

#     else:
#         return go.Figure(), html.Div()  # Return empty figure and empty div for the table

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)













# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# from datetime import datetime as dt
# import plotly.graph_objects as go  # Importing go instead of px for greater control
# import pandas as pd
# from mainMinVar import runFactorPortfolio3

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date - After or = 12/31/2009'),
#     dcc.DatePickerSingle(
#         id='date-1-picker',
#         date=dt.today()
#     ),
#     html.Label('End Date - Before or = 3/6/2023'),
#     dcc.DatePickerSingle(
#         id='date-2-picker',
#         date=dt.today()
#     ),
#     html.Label('Beta Lookback Options: 6 Months(Enter 126) - 1 Year(Enter 252) - 3 Year(Enter 756)'),
#     dcc.Input(
#         id='val1-input',
#         type='number',
#         value=0
#     ),
#     html.Label('Beta grouping options: Daily(Enter 1) - Monthly(Enter 28)'),
#     dcc.Input(
#         id='val2-input',
#         type='number',
#         value=0
#     ),
#     html.Button('Run', id='run-button', n_clicks=0),
#     dcc.Graph(id='graph-output')  # Graph component for displaying the graph
# ])

# # Define callback to run the function
# @app.callback(
#     Output('graph-output', 'figure'),  # Output for graph
#     [Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('date-1-picker', 'date'),
#      dash.dependencies.State('date-2-picker', 'date'),
#      dash.dependencies.State('val1-input', 'value'),
#      dash.dependencies.State('val2-input', 'value')]
# )
# def run_function(n_clicks, date_1, date_2, val1, val2):
#     if n_clicks > 0:
#         # Call your function with provided inputs
#         outputDF, portfolioXVals, portfolioYVals, marketXVals, marketYVals = runFactorPortfolio3(date_1, date_2, val1, val2)

#         # Prepare data for the graph
#         portfolio_df = pd.DataFrame({'Dates': portfolioXVals, 'Portfolio': portfolioYVals})
#         market_df = pd.DataFrame({'Dates': marketXVals, 'S&P': marketYVals})

#         # Create figure
#         fig = go.Figure()

#         # Add portfolio line
#         # fig.add_trace(go.Scatter(x=portfolioXVals, y=portfolioYVals, mode='lines', name='Portfolio'))

#         # # Add S&P line
#         # fig.add_trace(go.Scatter(x=marketXVals, y=marketYVals, mode='lines', name='S&P'))

#         # fig.update_layout(title='Portfolio vs S&P', xaxis_title='Date', yaxis_title='Value')
        
#         fig.add_trace(go.Scatter(x=marketXVals, y=marketYVals, mode='lines', name='Portfolio'))

#         # Add S&P line
#         fig.add_trace(go.Scatter(x=marketXVals, y=portfolioYVals, mode='lines', name='S&P'))

#         fig.update_layout(title='Portfolio vs S&P', xaxis_title='Date', yaxis_title='Value')
        
#         fig.update_yaxes(range=[0, 500])  # Set the bounds of the y-axis

#         return fig  # Return the figure

#     else:
#         return go.Figure()  # Return empty figure

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
























# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# from datetime import datetime as dt
# from minVarComputingFunctions import printVals  # Import your function from your_module
# from mainMinVar import runFactorPortfolio3
# from dash import dash_table
# import plotly.graph_objs as go


# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date - After or = 12/31/2009'),
#     dcc.DatePickerSingle(
#         id='date-1-picker',
#         date=dt.today()
#     ),
#     html.Label('End Date - Before or = 3/6/2023'),
#     dcc.DatePickerSingle(
#         id='date-2-picker',
#         date=dt.today()
#     ),
#     html.Label('Beta Lookback Options: 6 Months(Enter 126) - 1 Year(Enter 252) - 3 Year(Enter 756)'),
#     dcc.Input(
#         id='val1-input',
#         type='number',
#         value=0
#     ),
#     html.Label('Beta grouping options: Daily(Enter 1) - Monthly(Enter 28)'),
#     dcc.Input(
#         id='val2-input',
#         type='number',
#         value=0
#     ),
#     html.Button('Run', id='run-button', n_clicks=0),
#     dash_table.DataTable(id='output'),
#     dcc.Graph(id='graph-output')  # Graph component for displaying the graph
# ])

# # Define callback to run the function
# @app.callback(
#     [Output('output', 'data'), Output('graph-output', 'figure')],  # Multiple outputs for data table and graph
#     [Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('date-1-picker', 'date'),
#      dash.dependencies.State('date-2-picker', 'date'),
#      dash.dependencies.State('val1-input', 'value'),
#      dash.dependencies.State('val2-input', 'value')]
# )
# def run_function(n_clicks, date_1, date_2, val1, val2):
#     if n_clicks > 0:
#         # Call your function with provided inputs
#         # outputDF, portfolioXVals, yVals = runFactorPortfolio3(date_1, date_2, val1, val2)
#         outputDF, portfolioXVals, portfolioYVals, marketXVals, marketYVals = runFactorPortfolio3(date_1, date_2, val1, val2)

#         # Prepare data for the graph

#         # Create the graph
#         # graph_figure = {
#         #     'data': [
#         #         go.Scatter(x=['A','B','C'], y=[0,1,2], mode='lines+markers', name='Portfolio and')
#         #     ],
#         #     'layout': go.Layout(
#         #         title='Graph Title',
#         #         xaxis={'title': 'X Axis Title'},
#         #         yaxis={'title': 'Y Axis Title'}
#         #     )
#         # }
        
#         # graph_figure = {
#         #     'data': [
#         #         go.Scatter(x=portfolioXVals, y=portfolioYVals, mode='lines+markers', name='Price Data', line=dict(color='blue')),
#         #         go.Scatter(x=marketXVals, y=marketYVals, mode='lines+markers', name='Price Data', line=dict(color='red'))  # Add another trace

#         #     ],
#         #     'layout': go.Layout(
#         #         title='Portfolio(blue) vs S&P(red)',
#         #         xaxis={'title': 'Dates'},
#         #         yaxis={'title': 'Price'}
#         #     )
#         # }        
#         graph_figure = {
#     'data': [
#         go.Scatter(x=portfolioXVals, y=portfolioYVals, mode='lines+markers', name='Portfolio Price Data', line=dict(color='blue')),
#         go.Scatter(x=marketXVals, y=marketYVals, mode='lines+markers', name='S&P Price Data', line=dict(color='red'))
#     ],
#     'layout': go.Layout(
#         title='Portfolio vs S&P',
#         xaxis={'title': 'Dates'},
#         yaxis={'title': 'Price'}
#     )
# }


        
        
#         # Return data for both data table and graph
#         return outputDF.to_dict('records'), graph_figure
#     else:
#         return [], {}  # Return empty data and an empty graph if the button hasn't been clicked yet

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)





















# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# from datetime import datetime as dt
# from minVarComputingFunctions import printVals  # Import your function from your_module
# from mainMinVar import runFactorPortfolio3
# from dash import dash_table





# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date - After or = 12/31/2009'),
#     dcc.DatePickerSingle(
#         id='date-1-picker',
#         date=dt.today()
#     ),
#     html.Label('End Date - Before or = 3/6/2023'),
#     dcc.DatePickerSingle(
#         id='date-2-picker',
#         date=dt.today()
#     ),
#     html.Label('Beta Lookback Options: 6 Months(Enter 126) - 1 Year(Enter 252) - 3 Year(Enter 756)'),
#     dcc.Input(
#         id='val1-input',
#         type='number',
#         value=0
#     ),
#     html.Label('Beta grouping options: Daily(Enter 1) - Monthly(Enter 28)'),
#     dcc.Input(
#         id='val2-input',
#         type='number',
#         value=0
#     ),
#     html.Button('Run', id='run-button', n_clicks=0),
#     dash_table.DataTable(id='output')
# ])

# # Define callback to run the function
# @app.callback(
#     # Output('output', 'children'),
#     Output('output', 'data'),

#     [Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('date-1-picker', 'date'),
#      dash.dependencies.State('date-2-picker', 'date'),
#      dash.dependencies.State('val1-input', 'value'),
#      dash.dependencies.State('val2-input', 'value')]
# )
# def run_function(n_clicks, date_1, date_2, val1, val2):
#     if n_clicks > 0:
#         # result = printVals(date_1, date_2, val1, val2)  # Call your function with provided inputs
#         outputDF = runFactorPortfolio3(date_1, date_2, val1, val2)
#         return outputDF.to_dict('records')
#     else:
#         return []  # Return empty string if the button hasn't been clicked yet

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)







# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# from datetime import datetime as dt
# from minVarComputingFunctions import printVals  # Import your function from your_module
# from mainMinVar import runFactorPortfolio3







# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date - After or = 12/31/2009'),
#     dcc.DatePickerSingle(
#         id='date-1-picker',
#         date=dt.today()
#     ),
#     html.Label('End Date - Before or = 3/6/2023'),
#     dcc.DatePickerSingle(
#         id='date-2-picker',
#         date=dt.today()
#     ),
#     html.Label('Beta Lookback Options: 6 Months(Enter 126) - 1 Year(Enter 252) - 3 Year(Enter 756)'),
#     dcc.Input(
#         id='val1-input',
#         type='number',
#         value=0
#     ),
#     html.Label('Beta grouping options: Daily(Enter 1) - Monthly(Enter 28)'),
#     dcc.Input(
#         id='val2-input',
#         type='number',
#         value=0
#     ),
#     html.Button('Run', id='run-button', n_clicks=0),
#     html.Div(id='output')
# ])

# # Define callback to run the function
# @app.callback(
#     # Output('output', 'children'),
#     Output('output', 'children'),

#     [Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('date-1-picker', 'date'),
#      dash.dependencies.State('date-2-picker', 'date'),
#      dash.dependencies.State('val1-input', 'value'),
#      dash.dependencies.State('val2-input', 'value')]
# )
# def run_function(n_clicks, date_1, date_2, val1, val2):
#     if n_clicks > 0:
#         # result = printVals(date_1, date_2, val1, val2)  # Call your function with provided inputs
#         outputDF = runFactorPortfolio3(date_1, date_2, val1, val2)
#         return outputDF.to_dict('records')
#     else:
#         return ''  # Return empty string if the button hasn't been clicked yet

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)



















# import dash
# from dash import dcc, html
# from mainMinVar import runFactorPortfolio3

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date'),
#     dcc.Input(id='start-date', type='text', value='12/31/2009'),
#     html.Label('End Date'),
#     dcc.Input(id='end-date', type='text', value='3/6/2023'),
#     html.Label('Number of Lookback Days'),
#     dcc.Input(id='lookback-days', type='number', value=252),
#     html.Label('Beta Grouping Days'),
#     dcc.Input(id='beta-grouping-days', type='number', value=28),
#     html.Button('Run', id='run-button', n_clicks=0),
#     html.Div(id='output')
# ])

# # Define callback to run the program
# @app.callback(
#     dash.dependencies.Output('output', 'children'),
#     [dash.dependencies.Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('start-date', 'value'),
#      dash.dependencies.State('end-date', 'value'),
#      dash.dependencies.State('lookback-days', 'value'),
#      dash.dependencies.State('beta-grouping-days', 'value')]
# )
# def run_program(n_clicks, start_date, end_date, lookback_days, beta_grouping_days):
#     if n_clicks > 0:
#         # Here you can call your runFactorPortfolio3 function with the provided inputs
#         result = f'Start Date: {start_date}, End Date: {end_date}, Lookback Days: {lookback_days}, Beta Grouping Days: {beta_grouping_days}'
#         runFactorPortfolio3(start_date, end_date,lookback_days, beta_grouping_days )
#         return result

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
















# import dash
# from dash import dcc, html
# from mainMinVar import runFactorPortfolio3

# # Initialize Dash app
# app = dash.Dash(__name__)

# # Define layout
# app.layout = html.Div([
#     html.Label('Start Date'),
#     dcc.Input(id='start-date', type='text', value='12/31/2009'),
#     html.Label('End Date'),
#     dcc.Input(id='end-date', type='text', value='3/6/2023'),
#     html.Label('Number of Lookback Days'),
#     dcc.Input(id='lookback-days', type='number', value=252),
#     html.Label('Beta Grouping Days'),
#     dcc.Input(id='beta-grouping-days', type='number', value=28),
#     html.Button('Run', id='run-button', n_clicks=0),
#     html.Div(id='output')
# ])

# # Define callback to run the program
# @app.callback(
#     dash.dependencies.Output('output', 'children'),
#     [dash.dependencies.Input('run-button', 'n_clicks')],
#     [dash.dependencies.State('start-date', 'value'),
#      dash.dependencies.State('end-date', 'value'),
#      dash.dependencies.State('lookback-days', 'value'),
#      dash.dependencies.State('beta-grouping-days', 'value')]
# )
# def run_program(n_clicks, start_date, end_date, lookback_days, beta_grouping_days):
#     if n_clicks > 0:
#         # Here you can call your runFactorPortfolio3 function with the provided inputs
#         result = f'Start Date: {start_date}, End Date: {end_date}, Lookback Days: {lookback_days}, Beta Grouping Days: {beta_grouping_days}'
#         # output = runFactorPortfolio3(start_date, end_date, lookback_days, beta_grouping_days)
#         return result

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)























































# import shiny as ui
# from shiny.express import render
# from mainMinVar import runFactorPortfolio3

# # Define UI elements for user input
# start_date_input = ui.input_text("start_date", label="Start Date", value="12/31/2009")
# end_date_input = ui.input_text("end_date", label="End Date", value="3/6/2023")
# lookbackDaysInput = ui.input_number("num_days", label="6 Month(Enter 126), 1 Year(Enter 252), or 3 Year(Enter 756) Beta Lookback Days", min=1, max=10000, value=756)
# betaGroupingInput = ui.input_number("window_size", label="Daily(Enter 1) or Monthly(Enter 28) Beta Calculation", min=1, max=365, value=1)

# # Define function to run factor portfolio
# @render.text
# def run_factor_portfolio():
#     # Get user input values
#     start_date = start_date_input()
#     end_date = end_date_input()
#     lookbackDays = lookbackDaysInput()
#     betaGrouping = betaGroupingInput()

#     # Call the function with user input
#     result = runFactorPortfolio3(start_date, end_date, lookbackDays, betaGrouping)
    
#     # Return result (you may need to format this based on the return type of your function)
#     return result

# # Run the PyShiny application
# ui.render()
