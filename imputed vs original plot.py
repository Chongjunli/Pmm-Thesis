import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
import random


# function of dataset generation
def create_data(beta=1, sigma2=10, n=250):
    x = np.random.normal(size=n)
    y = beta * x + np.random.normal(scale=np.sqrt(sigma2), size=n)
    return pd.DataFrame({'x': x, 'y': y})


# # generate MCAR values
# def make_missing(data, p=0.5):
#     rx = np.random.binomial(n=1, p=p, size=data.shape[0])
#     data.loc[rx == 0, 'x'] = np.nan
#     return data

# generate MAR values (only the x values corresponding to the top 50% of y values
# will be randomly missing)
def make_missing(data, p=0.5):
    median_y = data['y'].median()
    rx = np.random.binomial(n=1, p=p, size=data.shape[0])
    data.loc[(data['y'] > median_y) & (rx == 0), 'x'] = np.nan
    return data


# generate dataset
data = create_data()
data_with_missing_values = make_missing(data.copy(), 0.5)

# define pmm method
def fill_missing_values(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predict_missing = model.predict(X_test)
    predict_obs = model.predict(X_train)

    # # PMM nearest neighbour
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing)
    #     closest_index = np.argmin(differences)  # 找到最接近的那个观测值的索引
    #     candidates.append(closest_index)  # 直接添加最接近的索引

    # # PMM top d candidates method (d=5)
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing)
    #     min_indices = np.argsort(differences)[:5]
    #     chosen_index = random.choice(min_indices.tolist())
    #     candidates.append(chosen_index)

    # PMM neighbors within a certain threshold (5)
    candidates = []
    for missing in predict_missing:
        differences = np.abs(predict_obs - missing) # Calculate absolute differences with all observed values
        valid_indices = np.where(differences < 100)[0] # Find valid indices where the difference is less than 5

        if len(valid_indices) > 0:
            chosen_index = random.choice(valid_indices.tolist()) # Randomly choose one valid index
            candidates.append(chosen_index)  # Add the selected index
        else:
            candidates.append(np.nan) # If no valid index is found, append NaN (or handle as needed)

    drawn_value = y_train.iloc[candidates]
    return np.array(drawn_value).flatten()


# a list for the storage of the imputed dataset
imputed_datasets = []

# generate multiple imputations
for dataset_num in range(5):
    # copy the imputed dataset
    data_imputed = data_with_missing_values.copy()

    # iteration of imputation
    for iteration in range(50):  # 迭代次数
        print(f"Dataset {dataset_num + 1}, Iteration {iteration + 1}...")

        # impute x values
        x_train_df = data_imputed.dropna(subset=['x'])
        x_test_df = data_imputed[data_imputed['x'].isna()]

        X_train = x_train_df[['y']]
        y_train = x_train_df['x']
        X_test = x_test_df[['y']]


        if len(X_test) > 0:
            x_filled_values = fill_missing_values(X_train, y_train, X_test)
            data_imputed.loc[data_imputed['x'].isna(), 'x'] = x_filled_values


    imputed_datasets.append(data_imputed)

# randomly choose a dataset from the multiple imputations
imputed_pmm = random.choice(imputed_datasets)

# IterativeImputer
imputer = IterativeImputer()
imputed_iterative = imputer.fit_transform(data_with_missing_values)

# SimpleImputer
simple_imputer = SimpleImputer(strategy='mean')
imputed_simple = simple_imputer.fit_transform(data_with_missing_values)

# convert to DataFrame for plotting
imputed_iterative = pd.DataFrame(imputed_iterative, columns=data.columns)
imputed_simple = pd.DataFrame(imputed_simple, columns=data.columns)

# set the figure
plt.figure(figsize=(6, 6))

# generate plots with different colors
plt.scatter(imputed_pmm['x'], imputed_pmm['y'], color='blue', label='PMM Imputation', alpha=0.6, s=66)
plt.scatter(imputed_iterative['x'], imputed_iterative['y'], color='green', label='Iterative Imputation', alpha=0.6, s=66)
plt.scatter(imputed_simple['x'], imputed_simple['y'], color='orange', label='Simple Imputation', alpha=0.6, s=66)
plt.scatter(data['x'], data['y'], color='red', label='Original Data', alpha=0.6, s=66)

# add label and title
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Different Imputation Methods')
plt.legend()
plt.grid()
plt.show()





















