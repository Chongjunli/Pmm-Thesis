import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from scipy.stats import norm
import random
import statsmodels.api as sm


# function of dataset generation
def create_data(beta=1, sigma2=1, n=500):
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

def fill_missing_values(X_train, y_train, X_test):
    # linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict missing values
    predict_missing = model.predict(X_test)
    predict_obs = model.predict(X_train)

    # # PMM nearest neighbours
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing)
    #     closest_index = np.argmin(differences)  # find the closest one
    #     candidates.append(closest_index)  # Add the selected index

    # # PMM top d candidates method (d=15 best)
    # candidates = []
    # for missing in predict_missing:
    #     differences = np.abs(predict_obs - missing)  # Calculate absolute differences with all observed values
    #     min_indices = np.argsort(differences)[:15]  # Get the indices of the closest 5
    #     chosen_index = random.choice(min_indices.tolist())  # Randomly choose one from the closest 5
    #     candidates.append(chosen_index)  # Add the selected index

    # PMM neighbors within a certain threshold (5)
    candidates = []
    for missing in predict_missing:
        differences = np.abs(predict_obs - missing) # Calculate absolute differences with all observed values
        valid_indices = np.where(differences < 1)[0] # Find valid indices where the difference is less than threshold

        if len(valid_indices) > 0:
            chosen_index = random.choice(valid_indices.tolist()) # Randomly choose one valid index
            candidates.append(chosen_index)  # Add the selected index
        else:
            candidates.append(np.nan) # If no valid index is found, append NaN (or handle as needed)

    drawn_value = y_train.iloc[candidates]
    return np.array(drawn_value).flatten()


def evaluate_imputation(imputed_data):
    # get the value of beta
    X = imputed_data[['x']]
    y = imputed_data['y']  # target variable
    model = LinearRegression()
    model.fit(X, y)

    estimate = model.coef_[0]  # estimate of beta
    true_value = 1  # set true value to 1

    # Raw bias and percent bias
    rb = estimate - true_value
    pb = 100 * abs(rb) / abs(true_value)


    # confidence interval
    se = np.std(y - model.predict(X)) / np.sqrt(len(X))  # standard error
    z_score = norm.ppf(1 - (1 - 0.95) / 2)
    lower_bound = estimate - z_score * se
    upper_bound = estimate + z_score * se

    # coverage rate
    cr = int(lower_bound <= true_value <= upper_bound)  # cover the true value: true/false

    # average width
    aw = upper_bound - lower_bound

    # RMSE
    rmse = (estimate - true_value) ** 2

    return rb, pb, cr, aw, rmse

# 10 times of simulation
def simulate_imputation(runs=10):
    rb_pmm_list, pb_pmm_list, cr_pmm_list, aw_pmm_list, rmse_pmm_list = [], [], [], [], []
    rb_iterative_list, pb_iterative_list, cr_iterative_list, aw_iterative_list, rmse_iterative_list = [], [], [], [], []
    rb_simple_list, pb_simple_list, cr_simple_list, aw_simple_list, rmse_simple_list = [], [], [], [], []

    for _ in range(runs):
        data = create_data()
        data_with_missing_values = make_missing(data.copy(), p=0.5)


        # PMM
        imputed_datasets = []
        for dataset_num in range(5):
            data_imputed = data_with_missing_values.copy()
            for iteration in range(50):
                x_train_df = data_imputed.dropna(subset=['x'])
                x_test_df = data_imputed[data_imputed['x'].isna()]

                X_train = x_train_df[['y']]
                y_train = x_train_df['x']
                X_test = x_test_df[['y']]

                if len(X_test) > 0:
                    x_filled_values = fill_missing_values(X_train, y_train, X_test)
                    data_imputed.loc[data_imputed['x'].isna(), 'x'] = x_filled_values

            imputed_datasets.append(data_imputed)

        selected_dataset = random.choice(imputed_datasets)

        # IterativeImputer
        imputer = IterativeImputer()
        imputed_iterative = imputer.fit_transform(data_with_missing_values)

        # SimpleImputer
        simple_imputer = SimpleImputer(strategy='mean')
        imputed_simple = simple_imputer.fit_transform(data_with_missing_values)


        # evaluate pmm
        rb_pmm, pb_pmm, cr_pmm, aw_pmm, rmse_pmm = evaluate_imputation(selected_dataset)
        rb_pmm_list.append(rb_pmm)
        pb_pmm_list.append(pb_pmm)
        cr_pmm_list.append(cr_pmm)
        aw_pmm_list.append(aw_pmm)
        rmse_pmm_list.append(rmse_pmm)

        # evaluate IterativeImputer
        rb_iterative, pb_iterative, cr_iterative, aw_iterative, rmse_iterative = evaluate_imputation(
            pd.DataFrame(imputed_iterative, columns=data.columns))
        rb_iterative_list.append(rb_iterative)
        pb_iterative_list.append(pb_iterative)
        cr_iterative_list.append(cr_iterative)
        aw_iterative_list.append(aw_iterative)
        rmse_iterative_list.append(rmse_iterative)

        # evaluate SimpleImputer
        rb_simple, pb_simple, cr_simple, aw_simple, rmse_simple = evaluate_imputation(
            pd.DataFrame(imputed_simple, columns=data.columns))
        rb_simple_list.append(rb_simple)
        pb_simple_list.append(pb_simple)
        cr_simple_list.append(cr_simple)
        aw_simple_list.append(aw_simple)
        rmse_simple_list.append(rmse_simple)

    # get the average values
    pmm_mean_results = {
        'RB': np.mean(rb_pmm_list),
        'PB': 100 * abs(np.mean(rb_pmm_list)),
        'CR': np.mean(cr_pmm_list),
        'AW': np.mean(aw_pmm_list),
        'RMSE': np.sqrt(np.mean(rmse_pmm_list))
    }

    iterative_mean_results = {
        'RB': np.mean(rb_iterative_list),
        'PB': 100 * abs(np.mean(rb_iterative_list)),
        'CR': np.mean(cr_iterative_list),
        'AW': np.mean(aw_iterative_list),
        'RMSE': np.sqrt(np.mean(rmse_iterative_list))
    }

    simple_mean_results = {
        'RB': np.mean(rb_simple_list),
        'PB': 100 * abs(np.mean(rb_simple_list)),
        'CR': np.mean(cr_simple_list),
        'AW': np.mean(aw_simple_list),
        'RMSE': np.sqrt(np.mean(rmse_simple_list))
    }

    # round the values by keeping 3 digits
    pmm_mean_results = {k: round(float(v), 3) for k, v in pmm_mean_results.items()}
    iterative_mean_results = {k: round(float(v), 3) for k, v in iterative_mean_results.items()}
    simple_mean_results = {k: round(float(v), 3) for k, v in simple_mean_results.items()}

    return pmm_mean_results, iterative_mean_results, simple_mean_results

# run the simulation
pmm_mean, iterative_mean, simple_mean = simulate_imputation(runs=100)

# print the result
print("Pmm Method Mean Results:", pmm_mean)
print("IterativeImputer Mean Results:", iterative_mean)
print("SimpleImputer Mean Results:", simple_mean)

