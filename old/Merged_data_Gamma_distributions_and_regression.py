import pandas as pd
import numpy as np
from scipy.stats import gamma, kstest, chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from io import StringIO
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Load the data
data = pd.read_excel('Input_Data/Merged_data.xlsx')

# Function to remove outliers within each group
def remove_outliers(group, df,column,  lower, upper):
    # define a list with unique group name

    Q1 = group[column].quantile(lower) #define the values in the lowest
    Q3 = group[column].quantile(upper) #define the value in the highest
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
    df = pd.concat([df, group])

    # Filter and return the group without outliers
    return df
# Prepare to store the results
groups = data['Group'].unique()
groups = sorted(groups)
gamma_params = pd.DataFrame(columns=['Group', 'Shape', 'Location', 'Scale'])
ks_results = pd.DataFrame(columns=['Group', 'KS Statistic', 'KS P-Value'])
chi_results = pd.DataFrame(columns=['Group', 'Chi Statistic', 'Chi P-Value'])
grouped_data = data.groupby('Group')
#create a new dataframe to store after removing outliers
data_filtered =pd.DataFrame()
# Iterate over each group and remove outliers
for group_name, group_data in grouped_data:
    data_filtered = remove_outliers(group_data, data_filtered, 'ORtime', 0.25, 0.75)
data = data_filtered
#setup a figure to store qq plots
fig, axes = plt.subplots(nrows=len(groups), ncols=2, figsize=(15, 6 * len(groups)))

for i, group in enumerate(groups):
    group_data = data[data['Group'] == group]['ORtime'].dropna()
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    # Histogram on the left
    sns.histplot(group_data, kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Histogram for Group {group}')

    # Boxplot on the right
    sns.boxplot(x=group_data, ax=axes[i, 1], color='blue')
    axes[i, 1].set_title(f'Boxplot for Group {group}')

    # Fit gamma distribution, fix location to 0
    params = gamma.fit(group_data, floc=0)
    row = pd.DataFrame({'Group': [group],'Shape': [params[0]],'Location': [params[1]],  'Scale': [params[2]]})
    gamma_params = pd.concat([gamma_params, row])

    # Perform and store KS test
    ks_stat, ks_p_value = kstest(group_data, 'gamma', args=(params[0], params[1], params[2]))
    row = pd.DataFrame({'Group': [group], 'KS Statistic': [ks_stat], 'KS P-Value': [ks_p_value]})
    ks_results = pd.concat([ks_results, row])

    # Perform and store Chi-squared test
    observed, bin_edges = np.histogram(group_data, bins='auto', density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Midpoints of bins
    expected = gamma.pdf(bin_centers, *params)  # Using the params from gamma fitting
    # Scaling the expected frequencies
    expected_scaled = expected * (observed.sum() / expected.sum())
    # Now use the scaled expected frequencies for the Chi-squared test
    chi_stat, chi_p_value = chisquare(observed, expected_scaled)
    row = pd.DataFrame({'Group': [group], 'Chi Statistic': [chi_stat], 'Chi P-Value': [chi_p_value]})
    chi_results = pd.concat([chi_results, row])

# Adjust layout
fig.tight_layout()
fig.savefig('group_boxplots_before_regression.png')
test_results = pd.merge(ks_results, chi_results, on='Group')

#create a column with averagere surgery duration per group to the parameter dataframe
average_duration_per_group = data.groupby('Group')['ORtime'].mean().reset_index()

#create a dataframe to store the parameters, averages and group and category number
group_data = data.groupby('Group').agg({'Category': 'first', 'Group_number': 'first'}).reset_index()
group_data = pd.merge(group_data, average_duration_per_group)
gamma_params = pd.merge(gamma_params, group_data, on='Group', how='left')

data = data.dropna(thresh=10, axis=1) #deleting the columns with less then 10 entries
def try_out (data, low, high):
    print(low, high)
    # create a new dataframe to store after removing outliers
    data_filtered =pd.DataFrame()
    grouped_data = data.groupby('Group')
    # Iterate over each group and remove outliers
    for group_name, group_data in grouped_data:
        data_filtered = remove_outliers(group_data, data_filtered, 'Wardtime', low,high)
    data = data_filtered
    # data = data[~((data['Group'] == 23) & (data['Wardtime'] > 3100))]

    print(len(data))
    # setup a figure to store qq plots
    fig, axes = plt.subplots(nrows=len(groups), ncols=3, figsize=(18, 6 * len(groups)))

    for i, group in enumerate(groups):
        group_data = data[data['Group'] == group].dropna(subset=['Wardtime', 'ORtime'])

        # Histogram on the left
        sns.histplot(group_data['Wardtime'], kde=True, ax=axes[i, 0], color='blue')
        axes[i, 0].set_title(f'Histogram for Group {group}')

        # Boxplot on the right
        sns.boxplot(x=group_data['Wardtime'], ax=axes[i, 1], color='blue')
        axes[i, 1].set_title(f'Boxplot for Group {group}')

        # Scatterplot
        sns.scatterplot(data=group_data, x='ORtime', y='Wardtime', ax=axes[i, 2], color='blue')
        axes[i, 2].set_title(f'Scatterplot for Group {group}')

    # Adjust layout
    fig.tight_layout()
    fig.savefig('group_regression_plots.png')

    # Create dummy variables for 'Category'
    dummies = pd.get_dummies(data['Group'], prefix='Group', drop_first= True) #
    # dummies.drop('Group_42', axis=1, inplace=True) #setting the reference category
    dummies = dummies.astype(int)

    # Concatenate dummies with the original DataFrame
    df = pd.concat([data, dummies], axis=1)

    # Define independent variables (including dummy variables and 'ORtime')
    X = df[dummies.columns.tolist()+ ['ORtime']]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Define dependent variable
    y = df['Wardtime']

    # Fit the regression model
    Ward_model = sm.OLS(y, X).fit()
    # Print the regression results
    # print(Ward_model.summary())
    print(f'The r-squared is {round(Ward_model.rsquared,4)}')
    # Get p-values
    p_values = Ward_model.pvalues

    # Count significant p-values
    significant_count = (p_values > 0.05).sum()

    print("Number of insignificant p-values:", significant_count)

    # results_html = Ward_model.summary().tables[1].as_html()
    # df_regression_results = pd.read_html(StringIO(results_html), header=0, index_col=0)[0]

# average_length_of_stay = data.groupby('Group')['Wardtime'].mean().reset_index()
# average_length_of_stay = average_length_of_stay.rename(columns={'Wardtime':'Average_Lenght_Of_Stay'})
# gamma_params = pd.merge(gamma_params, average_length_of_stay, on='Group', how='left')
#
# ward_columns = ['Ward', 'Ward2', 'Ward3', 'Ward4']
# ic_keywords = ['IC']
# data['Assigned_to_IC'] = data[ward_columns].apply(lambda row: any(ic in str(w) for w in row for ic in ic_keywords), axis=1)
#
# # Categories to exclude
# exclude_categories = ['DER', 'NEU', 'OOG', 'PLCH', 'TAN']
#
# # Calculate the percentage of IC visits per category
# percentage_ic_visits = (data.groupby('Group')['Assigned_to_IC'].sum() / data.groupby('Group')['Assigned_to_IC'].count()) * 100
#
# params = pd.merge(gamma_params, percentage_ic_visits, on='Group', how='left')

# #Optionally, save results to Excel
# with pd.ExcelWriter('Model_data/analysis_results.xlsx') as writer:
#     gamma_params.to_excel(writer, sheet_name='Parameters')
#     test_results.to_excel(writer, sheet_name='Test Results')
# df_regression_results.to_excel('Model_data/regression_summary.xlsx')
# data.to_excel('Model_data/With_outliers_removed.xlsx')

try_out(data, 0.25, 0.75)