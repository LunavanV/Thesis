import pandas as pd

parameters = pd.read_excel('Model_data/regression_summary.xlsx')
def get_ward_stay(group, surgery_duration):
    zeros_list = [0] * 50
    if group !=50:
        zeros_list[group-1] = 1
        zeros_list[49] = surgery_duration
    X = zeros_list
    # Extract coefficients from parameters DataFrame
    coefficients = parameters['coef'].values
    # Calculate prediction
    estimated_y = parameters['coef'].iloc[0]  # Initialize with intercept
    for i in range(len(X)):
        estimated_y += coefficients[i+1] * X[i]
    return(round(estimated_y))

