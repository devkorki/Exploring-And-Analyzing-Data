"""
ITC6002A1 - Exploring And Analyzing Data - Fall Term 2024
Prof. Dimitrios Milioris

Final Project
Team 1
Evangelos Aspiotis
Panagiotis Korkizoglou
Christos Liakopoulos
"""

import subprocess
import sys

# List of required packages
required_packages = ['requests', 'pandas', 'numpy', 'matplotlib']


def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Check and install missing packages
# install_packages(required_packages)  # !!!---> uncomment to install necessary packages

# -------------------------------------------------------------------------------------

import os
import requests
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import glob


# Custom functions
def download_raw_data():
    urls_for_download = [
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/aef8997e-3b0f-4b02-9f59-0751a6093936/download/aghiosnikolaos.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/0c3607b3-0af2-49e6-9f19-7296a140d7c3/download/alexandroupolis.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/a4175f2a-b2a4-4571-b2f4-047f2a27ce5d/download/argos.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/d3d9efb7-aa84-4da8-a2c9-cf65ab6734b4/download/athens.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/955ee850-63f5-412b-8601-0952e3b010ef/download/florina.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/263b89fd-99cb-4f4b-b61f-57030801026d/download/ioannina.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/fe95c37f-8fe4-4574-84cb-bab7920dec5a/download/kerkyra.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/266ef3ff-856b-4a45-a1a3-e63308ecdb4c/download/larissa.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/ee3c35c0-7a49-44e9-8404-399ab4532533/download/sparti.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/8d708b40-f954-4ead-aea4-0923a27f75e9/download/thessaloniki.csv',
        'https://data.climpact.gr/dataset/68a1c5c6-739c-4001-ac9c-6cb08c0b1085/resource/c1e7d641-f180-4f2c-a813-b03391ea366c/download/tinos.csv'
    ]

    for url in urls_for_download:
        filename = url.split('/''')[-1]
        save_path = 'data/' + filename
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            file.write(response.content)


def replace_na_with_moving_average(dataframe, window_size):
    # Replace '---' values with the average of the previous 'window_size' rows
    for row_idx in range(len(dataframe)):
        row = dataframe.iloc[row_idx]

        # If the temperature is NaN (originally '---'), calculate the average of the previous 'window_size' rows
        if pd.isna(row['Temperature']):
            # Calculate the range of previous 'window_size' rows
            start_idx = max(0, row_idx - window_size)  # Ensure we don't go below index 0
            previous_rows = dataframe.iloc[start_idx:row_idx]  # Select the previous 'window_size' rows

            # Calculate the average of the previous rows, excluding NaNs
            valid_temperatures = previous_rows['Temperature'].dropna()
            if not valid_temperatures.empty:
                # Replace the NaN with the average of valid temperatures
                dataframe.at[row_idx, 'Temperature'] = valid_temperatures.mean()

    return dataframe


def moving_average_forecast(data, window_size, forecast_days):
    """
    Forecast future values using moving average for a specified number of days
    Parameters:
    - data (list or pd.Series): The input historical data.
    - window_size (int): The size of the moving average window.
    - forecast_days (int): The number of future days to forecast.
    Returns:
    - full_forecast (list): Historical data with appended forecasted values.
    """
    # Convert data to a list if it's not already
    data = list(data)
    # Limit the window size to the length of the available data
    window_size = min(window_size, len(data))
    # Start with the existing data
    full_forecast = data.copy()
    # Forecast for the required number of days
    for _ in range(forecast_days):
        next_forecast = sum(full_forecast[-window_size:]) / window_size
        # Append the forecasted value
        full_forecast.append(next_forecast)
    return full_forecast


def weighted_moving_average_forecast(data: pd.Series, window_size: int, forecast_days: int, weights_list: list) -> list:
    """
    Calculate the weighted moving average and forecast future values.

    Parameters:
    - data: pd.Series - The historical data series.
    - window_size: int - Number of data points considered in the moving average.
    - forecast_days: int - Number of days to forecast.
    - weights: list - List of weights of length equal to window_size.

    Returns:
    - list: Weighted moving average and forecasted values.
    """
    # Check if weights length matches the window size
    if len(weights_list) != window_size:
        raise ValueError("Length of weights must equal the window size.")

    # Check if weights sum to 1
    if not abs(sum(weights_list) - 1.0) < 1e-8:
        raise ValueError("Weights must sum to 1.")

    # Initialize result list
    result = data.tolist()

    # Calculate the weighted moving average for the historical data
    if len(data) >= window_size:
        last_window = data[-window_size:].copy()
        for _ in range(forecast_days):
            forecast_value = sum(last_window * weights_list)
            result.append(forecast_value)
            last_window = pd.concat([last_window[1:], pd.Series([forecast_value])], ignore_index=True)

    return result


def exponential_smoothing(data, alpha_parameter, forecast_days):
    """
    Exponential Smoothing Forecasting function that keeps the historical data
    and appends the forecasted values.

    Parameters:
    - data: List or pandas Series of historical data points
    - alpha: Smoothing parameter (0 < alpha < 1)
    - forecast_periods: Number of periods to forecast into the future

    Returns:
    - result: pandas DataFrame with both historical data and forecasted values
    """
    # Convert data to a list if it's not already
    data = list(data)

    # Start with the existing data
    full_forecast = data.copy()

    # Forecast for the required number of days
    last_forecast = data[0]  # The first forecast is the first value of the data

    for t in range(1, len(data)):
        last_forecast = alpha_parameter * data[t - 1] + (1 - alpha_parameter) * last_forecast

    # Now forecast the required number of future days
    for _ in range(forecast_days):
        # Apply Exponential Smoothing formula for forecasted values
        last_forecast = alpha_parameter * data[-1] + (1 - alpha_parameter) * last_forecast
        full_forecast.append(last_forecast)

    return full_forecast


def calculate_mad(actual_column, forecast_column):
    errors = abs(actual_column - forecast_column)
    mad = errors.mean()
    return mad


def calculate_mse(actual_column, forecast_column):
    errors_squared = (actual_column - forecast_column) ** 2
    mse = errors_squared.mean()
    return mse


def calculate_mape(actual_column, forecast_column):
    percentage_errors = abs((actual_column - forecast_column) / actual_column) * 100
    mape = percentage_errors.mean()
    return mape


def linear_regression(a, b):
    # Add a column of ones for the intercept term
    A_b = np.c_[np.ones((X.shape[0], 1)), a]  # Add intercept term (bias)

    # Calculate the optimal parameters using the normal equation
    coefficients = np.linalg.inv(A_b.T.dot(A_b)).dot(A_b.T).dot(b)

    return coefficients


def create_monthly_dataset(filePath):
    dataframe = pd.read_csv(filePath, usecols=["Date", "T_mean"])
    dataframe.rename(columns={'T_mean': 'Temperature'}, inplace=True)

    # Convert column Date to datetime type
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    # Keep values for 2022 and 2023
    dataframe = dataframe[(dataframe['Date'] >= pd.Timestamp('2022-01-01')) & (dataframe['Date'] <= pd.Timestamp('2022-12-31'))]
    # Convert Temperaature column to float type
    dataframe['Temperature'] = pd.to_numeric(dataframe['Temperature'], errors='coerce')
    # Sort by date
    dataframe = dataframe.sort_values(by='Date')

    # Clean up of any inconsistencies, outliers, missing data etc.
    # Replace "---" with NaN using the window variable for the moving average
    dataframe = replace_na_with_moving_average(dataframe, window_na)
    # Fill null values, first backward and then forward
    dataframe['Temperature'] = dataframe['Temperature'].bfill().ffill()
    # Drop any duplicates (we know they are not any but good to have as a general tool)
    dataframe = dataframe.drop_duplicates(subset='Date')
    # Detect and handle outliers
    Quarter1 = dataframe['Temperature'].quantile(0.25)
    Quarter3 = dataframe['Temperature'].quantile(0.75)
    IQrange = Quarter3 - Quarter1
    lower_limit = Quarter1 - 1.5 * IQrange
    upper_limit = Quarter3 + 1.5 * IQrange
    dataframe_cleaned = dataframe[(dataframe['Temperature'] >= lower_limit) & (dataframe['Temperature'] <= upper_limit)]
    # Filter to keep only the first day of each month
    dataframe_cleaned = dataframe_cleaned[dataframe_cleaned['Date'].dt.is_month_start]
    return dataframe_cleaned


def forecast_and_evaluate(alpha, beta, gamma, dataframe):
    dataframe = dataframe.copy()
    season_length = 365
    # Prepare data
    data = dataframe['Temperature'].values  # Actual temperature values
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonality = np.zeros(n)
    forecast = np.zeros(n)

    # Initialize level, trend, and seasonality
    level[0] = data[0]
    trend[0] = data[1] - data[0]  # Approximate initial trend
    seasonality[:season_length] = 1  # Assume initial seasonality factors are 1
    forecast[0] = level[0] * seasonality[0]  # Use level and seasonality for the first forecast

    # Adaptive forecasting loop
    for t in range(1, n):
        if t >= season_length:
            # Compute forecast for time t
            forecast[t] = (level[t - 1] + trend[t - 1]) * seasonality[t - season_length]
        else:
            # Forecast without seasonality for the first cycle
            forecast[t] = level[t - 1] + trend[t - 1]

        # Update level, trend, and seasonality
        if t >= season_length:
            seasonality[t] = gamma * (data[t] / (level[t - 1] + trend[t - 1])) + (1 - gamma) * seasonality[t - season_length]
        else:
            seasonality[t] = 1  # Set seasonality to 1 for initialization

        level[t] = alpha * (data[t] / seasonality[t]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]

    # Adjust forecasts for seasonality
    dataframe['Forecast Trend & Seasonality'] = forecast
    return dataframe


def find_optimal_parameters(dataframe):
    best_mape = float('inf')  # Start with an infinitely large MAPE
    best_params = (0, 0, 0)  # Store the best combination of alpha, beta, gamma

    # Loop over possible values for alpha, beta, and gamma
    for alpha in np.arange(0, 1.1, 0.1):
        for beta in np.arange(0, 1.1, 0.1):
            for gamma in np.arange(0, 1.1, 0.1):
                # Evaluate MAPE for the current combination
                dataframe = forecast_and_evaluate(alpha, beta, gamma, dataframe)
                mape = calculate_mape(dataframe['Temperature'], dataframe['Forecast Trend & Seasonality'])

                # Update the best combination if the current MAPE is lower
                if mape < best_mape:
                    best_mape = mape
                    best_params = (alpha, beta, gamma)

    return best_params, best_mape


# Global Variables

# Window size for the moving average used in the fillna function
window_na = 4

''' 
Question A
Choose a data set of a quantity whose daily values over two years are publicly available.
[For example, this could be Covid19 cases, stock market data, oil / gas prices, meteo data, etc.].
Go over this data set and make sure to clean it up of any inconsistencies, outliers, missing data, etc.
'''

'''
Download the selected raw data
The measurements come from the network of automatic meteorological stations of the National Observatory of Athens/meteo.gr
'''
# download_raw_data()  # !!!---> uncomment to run function

# Load only the T_mean column (Mean Temperature) and rename it to Temperature
file_path = 'data/athens.csv'
df = pd.read_csv(file_path, usecols=["Date", "T_mean"])
df.rename(columns={'T_mean': 'Temperature'}, inplace=True)

# Convert column Date to datetime type
df['Date'] = pd.to_datetime(df['Date'])
# Keep values for 2022 and 2023
df = df[(df['Date'] >= pd.Timestamp('2022-01-01')) & (df['Date'] <= pd.Timestamp('2023-12-31'))]
# Convert Temperaature column to float type
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
# Sort by date
df = df.sort_values(by='Date')

# Clean up of any inconsistencies, outliers, missing data etc.
# Replace "---" with NaN using the window variable for the moving average
df = replace_na_with_moving_average(df, window_na)
# Fill null values, first backward and then forward
df['Temperature'] = df['Temperature'].bfill().ffill()
# Drop any duplicates (we know they are not any but good to have as a general tool)
df = df.drop_duplicates(subset='Date')
# Detect and handle outliers
Q1 = df['Temperature'].quantile(0.25)
Q3 = df['Temperature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['Temperature'] >= lower_bound) & (df['Temperature'] <= upper_bound)]

'''
Question B
Using the 1st year data of your chosen (and cleaned up ) data set, try to forecast the data of months 2-12 based on the first month’s data.
'''

# Filter data to include only first year
first_year = df_cleaned['Date'].dt.year.min()
first_year_data = df_cleaned[df_cleaned['Date'].dt.year == first_year]
# Extract January’s data
january_data = first_year_data[first_year_data['Date'].dt.month == 1]
january_temps = january_data['Temperature']
# Extract February–December for evaluation
forecast_data = first_year_data[first_year_data['Date'].dt.month > 1]
forecast_temps = forecast_data['Temperature']

'''
Question B.a
Use 3 different forecasting techniques to perform the forecasting of level (without trend or seasonality adjustments).
If a technique is parametric, try to choose the optimal value for this parameter.
'''
# Calculate the number of forecast days
days_to_forecast = forecast_data.shape[0]

# Technique 1: Moving Average (MA) Method

# Forecast using the custom moving average function
# Initialize variables to track the minimum MAPE and the corresponding parameter
min_mape_ma = float('inf')  # Start with a very large value
best_window_ma = None
for param in range(2, 32, 1):
    whole_year_forecast_ma = moving_average_forecast(january_temps, param, days_to_forecast)
    # Add the temporary forecasts next to the actual values for comparison purposes
    first_year_data = first_year_data.copy()
    first_year_data.loc[:, 'forecast_ma'] = whole_year_forecast_ma[:len(first_year_data)]
    # Calculate temporary metric
    mape_ma = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_ma'])
    # Check if the current MAPE is the smallest
    if mape_ma < min_mape_ma:
        min_mape_ma = mape_ma
        best_window_ma = param

print(f'Moving Average Method: The optimal window is {best_window_ma} with minimum MAPE: {min_mape_ma:.2f}%')

# Run the forecast again using the optimal window size
whole_year_forecast_ma = moving_average_forecast(january_temps, best_window_ma, days_to_forecast)
# Add the optimal forecasts next to the actual values for comparison purposes
first_year_data = first_year_data.copy()
first_year_data.loc[:, 'forecast_ma'] = whole_year_forecast_ma[:len(first_year_data)]
# Calculate metrics
mad_ma = calculate_mad(first_year_data['Temperature'], first_year_data['forecast_ma'])
mse_ma = calculate_mse(first_year_data['Temperature'], first_year_data['forecast_ma'])
mape_ma = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_ma'])

# Technique 2: Weighted Moving Average (WMA) Method
# Forecast using the custom weighted moving average function

# Initialize variables to track the minimum MAPE and the corresponding parameters
# Define the step size for weights
step = 0.1
min_mape = float('inf')
optimal_weights = None
optimal_weights_list = []
best_window_wma = None
for window in range(2, 6):
    weights_grid = product(np.arange(0, 1 + step, step), repeat=window)
    for weights in weights_grid:
        if abs(sum(weights) - 1.0) < 1e-8:  # Ensure weights sum to 1
            whole_year_forecast_wma = weighted_moving_average_forecast(january_temps, window, days_to_forecast, weights)
            first_year_data['forecast_wma'] = whole_year_forecast_wma[:len(first_year_data)]
            mape_wma = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_wma'])
            if mape_wma < min_mape:
                min_mape = mape_wma
                optimal_weights = weights
                best_window_wma = window
                optimal_weights_list = [float(weight) for weight in optimal_weights]
print(
    f'Weighted Moving Average Method: The optimal weights are: {optimal_weights_list}, optimal window is: {best_window_wma} with minimum MAPE: {min_mape:.2f}%')

# Run the forecast again using the optimal window size and optimal weights
optimal_weights = [0.1, 0.2, 0.7]
whole_year_forecast_wma = weighted_moving_average_forecast(january_temps, 3, days_to_forecast, optimal_weights)
# Add the optimal forecasts next to the actual values for comparison purposes
first_year_data = first_year_data.copy()
first_year_data.loc[:, 'forecast_wma'] = whole_year_forecast_wma[:len(first_year_data)]
mad_wma = calculate_mad(first_year_data['Temperature'], first_year_data['forecast_wma'])
mse_wma = calculate_mse(first_year_data['Temperature'], first_year_data['forecast_wma'])
mape_wma = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_wma'])
# Set the Date as the index for better plotting
first_year_data.set_index('Date', inplace=True)

# Technique 3: Exponential Smoothing
# Forecast using the custom exponential smoothing function

# Initialize variables to track the minimum MAPE and the corresponding parameters
# Define the step size for alpha
min_mape_es = float('inf')
best_alpha_es = None
for param in np.arange(0.1, 1.1, 0.1):
    whole_year_forecast_es = exponential_smoothing(january_temps, param, days_to_forecast)
    # Add the temporary forecasts next to the actual values for comparison purposes
    first_year_data = first_year_data.copy()
    first_year_data.loc[:, 'forecast_es'] = whole_year_forecast_es[:len(first_year_data)]
    # Calculate temporary metric
    mape_es = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_es'])
    # Check if the current MAPE is the smallest
    if mape_es < min_mape_es:
        min_mape_es = mape_es
        best_alpha_es = param

print(f'Exponential Smoothing Method: The optimal alpha is {best_alpha_es} with minimum MAPE: {min_mape_es:.2f}%')

# Run the forecast again using the optimal window size and optimal weights
whole_year_forecast_es = exponential_smoothing(january_temps, best_alpha_es, days_to_forecast)
# Add the optimal forecasts next to the actual values for comparison purposes
first_year_data = first_year_data.copy()
first_year_data.loc[:, 'forecast_es'] = whole_year_forecast_es[:len(first_year_data)]
mad_es = calculate_mad(first_year_data['Temperature'], first_year_data['forecast_es'])
mse_es = calculate_mse(first_year_data['Temperature'], first_year_data['forecast_es'])
mape_es = calculate_mape(first_year_data['Temperature'], first_year_data['forecast_es'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(first_year_data.index, first_year_data['Temperature'], label='Actual Temperature', color='blue')
plt.plot(first_year_data.index, first_year_data['forecast_ma'], label='Forecast with MA', color='green')
plt.plot(first_year_data.index, first_year_data['forecast_wma'], label='Forecast with WMA', color='red')
plt.plot(first_year_data.index, first_year_data['forecast_es'], label='Forecast with ES', color='yellow')

plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual Temperature and Forecasts Over Time')
plt.legend()

plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''

Question B.c
Try to see if there is a trend in the 1st year’s data set, estimate it and then do a trend-adjusted forecasting.
'''

# Identify Trend
first_year_data = first_year_data.copy()
first_year_data['Day'] = np.arange(1, len(first_year_data) + 1)  # Time index for January
X = first_year_data[['Day']]  # Day index as feature
y = first_year_data['Temperature']  # Target variable

# Fit Linear Regression Model
theta = linear_regression(X, y)
# Calculate the predicted values (Trend)
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
first_year_data['Trend'] = X_b.dot(theta)  # Compute predictions
# Output the parameters (intercept and slope)
intercept, slope = theta
print(f"The parameters are: Intercept: {intercept} and Slope: {slope}")
#
# Compute MAD, MSE, and MAPE using the custom functions
mad_trend = calculate_mad(first_year_data['Temperature'], first_year_data['Trend'])
mse_trend = calculate_mse(first_year_data['Temperature'], first_year_data['Trend'])
mape_trend = calculate_mape(first_year_data['Temperature'], first_year_data['Trend'])

# Print the results
print(f"MAD (Mean Absolute Deviation): {mad_trend:.2f}")
print(f"MSE (Mean Squared Error): {mse_trend:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_trend:.2f}%")
first_year_data = first_year_data.reset_index()
# Plot first year Data with Trend
plt.figure(figsize=(10, 6))
plt.plot(first_year_data['Date'], first_year_data['Temperature'], label="Year Temperatures")
plt.plot(first_year_data['Date'], first_year_data['Trend'], label="Trend (Linear)", linestyle="--")
plt.legend()
plt.title("Year Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()

'''
Question B.d
Then try to see if there is seasonality in the 1st year’s data set, estimate the seasonality factor
and then adjust your forecasting for seasonality as well.
'''

# run the optimization custom function to get the optimal parameters using minimum MAPE as a criterion
optimal_params, min_mape = find_optimal_parameters(first_year_data)
print(f'Optimal parameters for Athens daily data are: Alpha={optimal_params[0]:.2f}, Beta={optimal_params[1]:.2f}, Gamma={optimal_params[2]:.2f}')
print(f"Minimum MAPE: {min_mape:.2f}%")

# use the optimal parameters found in the previous step to save forecast values to our dataframe
first_year_data = forecast_and_evaluate(optimal_params[0], optimal_params[1], optimal_params[2], first_year_data)

# Compute MAD, MSE, and MAPE using the custom functions
mad_trend_season = calculate_mad(first_year_data['Temperature'], first_year_data['Forecast Trend & Seasonality'])
mse_trend_season = calculate_mse(first_year_data['Temperature'], first_year_data['Forecast Trend & Seasonality'])
mape_trend_season = calculate_mape(first_year_data['Temperature'], first_year_data['Forecast Trend & Seasonality'])

# Print the results
print(f"MAD (Mean Absolute Deviation): {mad_trend_season:.2f}")
print(f"MSE (Mean Squared Error): {mse_trend_season:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_trend_season:.2f}%")

# Plot the original data and forecasts
plt.figure(figsize=(12, 6))
plt.plot(first_year_data['Date'], first_year_data['Temperature'], label='Actual Temperature', color='blue')
plt.plot(first_year_data['Date'], first_year_data['Forecast Trend & Seasonality'], label='Trend Adjusted Forecast with Seasonality', color='red', linestyle='dashed')
plt.title('Adaptive Forecasting with Trend and Seasonality')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

''' 
Question C
Using the 2nd year data of your chosen (and cleaned up ) data set, try to forecast the data of months 2-12 based on the first month’s data.
'''

# Filter data to include only the second year
second_year = df_cleaned['Date'].dt.year.min() + 1
second_year_data = df_cleaned[df_cleaned['Date'].dt.year == second_year]

# Extract January’s data
january_data_second = second_year_data[second_year_data['Date'].dt.month == 1]
january_temps_second = january_data_second['Temperature']

# Extract February–December for evaluation
forecast_data_second = second_year_data[second_year_data['Date'].dt.month > 1]
forecast_temps_second = forecast_data_second['Temperature']

'''
Question C.a 
Use 3 different forecasting techniques to perform the forecasting of level (without trend or seasonality adjustments).
If a technique is parametric, try to choose the optimal value for this parameter.
'''
# Calculate the number of forecast days
days_to_forecast_second = forecast_data_second.shape[0]

# Technique 1: Moving Average (MA) Method

# Forecast using the custom moving average function
# Initialize variables to track the minimum MAPE and the corresponding parameter
min_mape_ma_second = float('inf')  # Start with a very large value
best_window_ma_second = None
for param in range(2, 32, 1):
    whole_year_forecast_ma_second = moving_average_forecast(january_temps_second, param, days_to_forecast_second)
    second_year_data = second_year_data.copy()
    second_year_data.loc[:, 'forecast_ma'] = whole_year_forecast_ma_second[:len(second_year_data)]
    mape_ma_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_ma'])
    if mape_ma_second < min_mape_ma_second:
        min_mape_ma_second = mape_ma_second
        best_window_ma_second = param

print(f'Moving Average Method: The optimal window is {best_window_ma_second} with minimum MAPE: {min_mape_ma_second:.2f}%')

whole_year_forecast_ma_second = moving_average_forecast(january_temps_second, best_window_ma_second, days_to_forecast_second)
second_year_data = second_year_data.copy()
second_year_data.loc[:, 'forecast_ma'] = whole_year_forecast_ma_second[:len(second_year_data)]
mad_ma_second = calculate_mad(second_year_data['Temperature'], second_year_data['forecast_ma'])
mse_ma_second = calculate_mse(second_year_data['Temperature'], second_year_data['forecast_ma'])
mape_ma_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_ma'])

# Technique 2: Weighted Moving Average (WMA) Method
# Forecast using the custom weighted moving average function

# Initialize variables to track the minimum MAPE and the corresponding parameters
# Define the step size for weights
step = 0.1
min_mape_second = float('inf')
optimal_weights_second = None
optimal_weights_list_second = []
best_window_wma_second = None
for window in range(2, 6):
    weights_grid = product(np.arange(0, 1 + step, step), repeat=window)
    for weights in weights_grid:
        if abs(sum(weights) - 1.0) < 1e-8:
            whole_year_forecast_wma_second = weighted_moving_average_forecast(january_temps_second, window, days_to_forecast_second, weights)
            second_year_data['forecast_wma'] = whole_year_forecast_wma_second[:len(second_year_data)]
            mape_wma_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_wma'])
            if mape_wma_second < min_mape_second:
                min_mape_second = mape_wma_second
                optimal_weights_second = weights
                best_window_wma_second = window
                optimal_weights_list_second = [float(weight) for weight in optimal_weights_second]

print(f'Weighted Moving Average Method: The optimal weights are: {optimal_weights_list_second}, optimal window is: {best_window_wma_second} with minimum MAPE: {min_mape_second:.2f}%')

optimal_weights_second = [0.1, 0.2, 0.7]
whole_year_forecast_wma_second = weighted_moving_average_forecast(january_temps_second, 3, days_to_forecast_second, optimal_weights_second)
second_year_data = second_year_data.copy()
second_year_data.loc[:, 'forecast_wma'] = whole_year_forecast_wma_second[:len(second_year_data)]
mad_wma_second = calculate_mad(second_year_data['Temperature'], second_year_data['forecast_wma'])
mse_wma_second = calculate_mse(second_year_data['Temperature'], second_year_data['forecast_wma'])
mape_wma_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_wma'])

# Technique 3: Exponential Smoothing
# Forecast using the custom exponential smoothing function

# Initialize variables to track the minimum MAPE and the corresponding parameters
# Define the step size for alpha
min_mape_es_second = float('inf')
best_alpha_es_second = None
for param in np.arange(0.1, 1.1, 0.1):
    whole_year_forecast_es_second = exponential_smoothing(january_temps_second, param, days_to_forecast_second)
    second_year_data = second_year_data.copy()
    second_year_data.loc[:, 'forecast_es'] = whole_year_forecast_es_second[:len(second_year_data)]
    mape_es_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_es'])
    if mape_es_second < min_mape_es_second:
        min_mape_es_second = mape_es_second
        best_alpha_es_second = param

print(f'Exponential Smoothing Method: The optimal alpha is {best_alpha_es_second} with minimum MAPE: {min_mape_es_second:.2f}%')

whole_year_forecast_es_second = exponential_smoothing(january_temps_second, best_alpha_es_second, days_to_forecast_second)
second_year_data = second_year_data.copy()
second_year_data.loc[:, 'forecast_es'] = whole_year_forecast_es_second[:len(second_year_data)]
mad_es_second = calculate_mad(second_year_data['Temperature'], second_year_data['forecast_es'])
mse_es_second = calculate_mse(second_year_data['Temperature'], second_year_data['forecast_es'])
mape_es_second = calculate_mape(second_year_data['Temperature'], second_year_data['forecast_es'])

plt.figure(figsize=(10, 6))
plt.plot(second_year_data['Date'], second_year_data['Temperature'], label='Actual Temperature', color='blue')
plt.plot(second_year_data['Date'], second_year_data['forecast_ma'], label='Forecast with MA', color='green')
plt.plot(second_year_data['Date'], second_year_data['forecast_wma'], label='Forecast with WMA', color='red')
plt.plot(second_year_data['Date'], second_year_data['forecast_es'], label='Forecast with ES', color='yellow')

plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual Temperature and Forecasts Over Time (Second Year)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''

Question C.c
Try to see if there is a trend in the 1st year’s data set, estimate it and then do a trend-adjusted forecasting.
'''

# Identify Trend
second_year_data = second_year_data.copy()
second_year_data['Day'] = np.arange(1, len(second_year_data) + 1)
X_second = second_year_data[['Day']]
y_second = second_year_data['Temperature']
theta_second = linear_regression(X_second, y_second)
X_b_second = np.c_[np.ones((X_second.shape[0], 1)), X_second]
second_year_data['Trend'] = X_b_second.dot(theta_second)
intercept_second, slope_second = theta_second
print(f"The parameters are: Intercept: {intercept_second} and Slope: {slope_second}")

mad_trend_second = calculate_mad(second_year_data['Temperature'], second_year_data['Trend'])
mse_trend_second = calculate_mse(second_year_data['Temperature'], second_year_data['Trend'])
mape_trend_second = calculate_mape(second_year_data['Temperature'], second_year_data['Trend'])

print(f"MAD (Mean Absolute Deviation): {mad_trend_second:.2f}")
print(f"MSE (Mean Squared Error): {mse_trend_second:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_trend_second:.2f}%")
second_year_data = second_year_data.reset_index()

plt.figure(figsize=(10, 6))
plt.plot(second_year_data['Date'], second_year_data['Temperature'], label="Year Temperatures")
plt.plot(second_year_data['Date'], second_year_data['Trend'], label="Trend (Linear)", linestyle="--")
plt.legend()
plt.title("Year Temperature Trend (Second Year)")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.show()

'''
Question C.d
Then try to see if there is seasonality in the 1st year’s data set, estimate the seasonality factor
and then adjust your forecasting for seasonality as well.
'''

optimal_params_second, min_mape_second = find_optimal_parameters(second_year_data)
print(f'Optimal parameters for the second year data are: Alpha={optimal_params_second[0]:.2f}, Beta={optimal_params_second[1]:.2f}, Gamma={optimal_params_second[2]:.2f}')
print(f"Minimum MAPE: {min_mape_second:.2f}%")

second_year_data = forecast_and_evaluate(optimal_params_second[0], optimal_params_second[1], optimal_params_second[2], second_year_data)

mad_trend_season_second = calculate_mad(second_year_data['Temperature'], second_year_data['Forecast Trend & Seasonality'])
mse_trend_season_second = calculate_mse(second_year_data['Temperature'], second_year_data['Forecast Trend & Seasonality'])
mape_trend_season_second = calculate_mape(second_year_data['Temperature'], second_year_data['Forecast Trend & Seasonality'])

print(f"MAD (Mean Absolute Deviation): {mad_trend_season_second:.2f}")
print(f"MSE (Mean Squared Error): {mse_trend_season_second:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_trend_season_second:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(second_year_data['Date'], second_year_data['Temperature'], label='Actual Temperature', color='blue')
plt.plot(second_year_data['Date'], second_year_data['Forecast Trend & Seasonality'], label='Trend Adjusted Forecast with Seasonality', color='red', linestyle='dashed')
plt.title('Adaptive Forecasting with Trend and Seasonality (Second Year)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

'''
Question E 
Finally, choose a monthly data set (i.e. only 12 data points) from either the 1st or the 2nd year of your data (at regular intervals, 
e.g. the 1st day of each month). 
Then find at least another 10 similar data sets from other sources. 
For example, if your data set is the peak temperature of a municipality of Athens on the 1st of each month, 
find another 10 Athens municipalities that you expect to have similar weather, and collect the corresponding data. 
Run your best forecasting technique for the 12 point data set. 
Then adjust it taking advantage of the other 10 similar data sets.
'''

# Get the list of all CSV files in the 'data' folder
cities = glob.glob('data/*.csv')

# for all city files run an optimization to get the optimal parameters using minimum MAPE as a criterion
for city in cities:
    optimal_params, min_mape = find_optimal_parameters((create_monthly_dataset(city)))
    city = city.replace("data\\", "").replace(".csv", "").capitalize()  # Remove "data\", ".csv" and capitalize to get only city name
    print(f'Optimal parameters for {city} monthly data are: Alpha={optimal_params[0]:.2f}, Beta={optimal_params[1]:.2f}, Gamma={optimal_params[2]:.2f}')
    print(f"Minimum MAPE: {min_mape:.2f}%")