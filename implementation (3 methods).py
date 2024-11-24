import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random


pd.set_option('display.max_rows', None)

# Load data
airquality = pd.read_excel("C:/Users/lenovo/Desktop/thesis/airquality.xlsx")
airquality.columns = ['Ozone', 'Solar.R', 'Wind', 'Temp', 'Month', 'Day']

# Create a list to store the imputed datasets
imputed_datasets = []

# Define the function to fill in missing values
def fill_missing_values(X_train, y_train, X_test):
    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict missing values
    predict_missing = model.predict(X_test)
    predict_obs = model.predict(X_train)

    # PMM top d candidates method (d=5)
    candidates = []
    for missing in predict_missing:
        differences = np.abs(predict_obs - missing) # Calculate absolute differences with all observed values
        min_indices = np.argsort(differences)[:5]  # Get the indices of the closest 5
        chosen_index = random.choice(min_indices.tolist()) # Randomly choose one from the closest 5

        candidates.append(chosen_index)  # Add the selected index


    # # PMM nearest neighbour
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing)
    #     closest_index = np.argmin(differences)  # Get the closest one
    #     candidates.append(closest_index)


    # # PMM neighbors within a certain threshold (5)
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing) # Calculate absolute differences with all observed values
    #     valid_indices = np.where(differences < 5)[0] # Find valid indices where the difference is less than 5
    #
    #     if len(valid_indices) > 0:
    #         chosen_index = random.choice(valid_indices.tolist()) # Randomly choose one valid index
    #         candidates.append(chosen_index)  # Add the selected index
    #     else:
    #         candidates.append(np.nan) # If no valid index is found, append NaN (or handle as needed)

    drawn_value = y_train.iloc[candidates]
    return np.array(drawn_value).flatten()

# Generate multiple imputed datasets
for dataset_num in range(5):
    # Create a new DataFrame to store the imputed data
    airquality_imputed = airquality.copy()

    # Start iterating to fill missing values
    for iteration in range(50):  # Number of iterations
        print(f"Dataset {dataset_num + 1}, Iteration {iteration + 1}...")

        # Step 1: Fill in the Solar.R column
        solar_train_df = airquality_imputed.dropna(subset=['Solar.R'])
        solar_test_df = airquality_imputed[airquality_imputed['Solar.R'].isna()]

        X_train_solar = solar_train_df[['Wind', 'Temp', 'Month', 'Day']]
        y_train_solar = solar_train_df['Solar.R']
        X_test_solar = solar_test_df[['Wind', 'Temp', 'Month', 'Day']]

        # Fill in missing Solar.R values
        if len(X_test_solar) > 0:
            solar_filled_values = fill_missing_values(X_train_solar, y_train_solar, X_test_solar)
            airquality_imputed.loc[airquality_imputed['Solar.R'].isna(), 'Solar.R'] = solar_filled_values

        # Step 2: Fill in the Ozone column
        ozone_train_df = airquality_imputed.dropna(subset=['Ozone'])
        ozone_test_df = airquality_imputed[airquality_imputed['Ozone'].isna()]

        X_train_ozone = ozone_train_df[['Solar.R', 'Wind', 'Temp', 'Month', 'Day']]
        y_train_ozone = ozone_train_df['Ozone']
        X_test_ozone = ozone_test_df[['Solar.R', 'Wind', 'Temp', 'Month', 'Day']]

        # Fill in missing Ozone values
        if len(X_test_ozone) > 0:
            ozone_filled_values = fill_missing_values(X_train_ozone, y_train_ozone, X_test_ozone)
            airquality_imputed.loc[airquality_imputed['Ozone'].isna(), 'Ozone'] = ozone_filled_values

    # Add the imputed dataset to the list
    imputed_datasets.append(airquality_imputed)

# Randomly select one imputed dataset
selected_dataset = random.choice(imputed_datasets)

# Display the imputed data
print(selected_dataset)



