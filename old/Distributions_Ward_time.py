import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, gamma, weibull_min, norm, fisk, invgamma, chisquare, kstest  # fisk is log-logistic in SciPy
#setting a randomly chosen seed in order to create reproducability
np.random.seed(50)

#importing all functions needed to run the code
from Functions_distributions import fit_distribution, plot_qq, chi_squared_test, ks_test, remove_outliers

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_excel('../Input_Data/df_total.xlsx')
distribution_parameters_surgery_duration = pd.read_excel('../Model_data/Distributie_parameters_only_surgery_duration.xlsx')
new_data = distribution_parameters_surgery_duration[['Group', 'Category', 'Group number']]
data = data.merge(new_data, how='left', on=['Group number', 'Category'])
groups = data['Group'].unique()
groups = sorted(groups)
# Setting the style for the plots
sns.set(style="whitegrid")

data = data.rename(columns={'Total_Time_Ward':'Ward1_time'})
# Extract ward number from the column name
data['Total_Time_Ward'] = data['Ward1_time'].fillna(0) + data['Ward2_time'].fillna(0) + data['Ward3_time'].fillna(0) + data['Ward4_time'].fillna(0)
data = data.dropna(thresh=10, axis=1) #deleting the columns with less then 10 entries

data = data[data['Total_Time_Ward'] >= 0]
data = data[data['Total_Time_Ward'] <= 20000]

sns.histplot(data['Total_Time_Ward'], kde=True, bins='auto', color='blue')
plt.title('Histogram of Total Ward Stay')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Total_Time_Ward'], color='lightblue')
plt.title('Boxplot of Total Ward Stay')
plt.ylabel('Duration in Minutes')
plt.savefig('boxplot_surgery.png')
plt.show()

data_filtered =pd.DataFrame()
grouped_data = data.groupby('Group')
# Iterate over each group and remove outliers
for group_name, group_data in grouped_data:
    filtered_group = remove_outliers(group_data, 'Total_Time_Ward', 0.2, 0.75)
    data_filtered = pd.concat([data_filtered, filtered_group], ignore_index=True)
data = data_filtered

sns.histplot(data['Total_Time_Ward'], kde=True, bins='auto', color='blue')
plt.title('Histogram of Total Ward Stay')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Total_Time_Ward'], color='lightblue')
plt.title('Boxplot of Total Ward Stay')
plt.ylabel('Duration in Minutes')
plt.savefig('boxplot_surgery.png')
plt.show()

# Assuming 'data' and 'groups' are predefined
fig, axes = plt.subplots(nrows=50, ncols=5, figsize=(30, 150))  # Large figure size to accommodate all subplots
distributions = ['gamma', 'weibull', 'lognorm','log-logistic',  'pearsonV']

all_results = pd.DataFrame()

# Iterate over each group to perform analyses and plotting
for group_index, group in enumerate(groups):
    group_data = data[data['Group'] == group]['Total_Time_Ward']
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    # Dictionary to store results for this category
    results = {}

    for i, dist in enumerate(distributions):
        # Fit distribution and perform tests
        params = fit_distribution(group_data, dist)
        # Generate QQ plot in the assigned subplot
        ax = axes[group_index, i]  # Select the appropriate subplot
        plot_qq(group_data, dist, params, ax)
        ax.set_title(f'{dist.capitalize()} (Group: {group})')

        chi2_result = chi_squared_test(group_data, dist, params)
        ks_result = ks_test(group_data, dist, params)

        # Store results
        row = pd.DataFrame({
            'distribution': [dist],
            'Group': [group],
            'Parameters': [params],
            'Chi-Squared Statistic': [chi2_result.statistic],
            'Chi-Squared P-value': [chi2_result.pvalue],
            'KS Statistic': [ks_result.statistic],
            'KS P-value': [ks_result.pvalue]
        })
        all_results = pd.concat([all_results, row])

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('qq_plots_ward_stay.png')

chi_counts = all_results.groupby('distribution').agg(
    ChiSquared_001=('Chi-Squared P-value', lambda x: (x > 0.01).sum()),
    ChiSquared_005=('Chi-Squared P-value', lambda x: (x > 0.05).sum())
).reset_index()

# Filter rows where 'KS P-value' is greater than 0.01 and 0.05
ks_counts = all_results.groupby('distribution').agg(
    KS_001=('KS P-value', lambda x: (x > 0.01).sum()),
    KS_005=('KS P-value', lambda x: (x > 0.05).sum())
).reset_index()

# Merge the counts
overview = pd.merge(chi_counts, ks_counts, on='distribution', how='outer')

# Fill missing values with 0
overview = overview.fillna(0)

# Group by 'Group' and find the minimum KS P-value
max_ks_pvalues = all_results.groupby('Group')['KS P-value'].max().reset_index()

# Merge to get all distributions with the minimum KS P-value for each group
highest_ks_pvalues = pd.merge(all_results, max_ks_pvalues, on=['Group', 'KS P-value'])
# Group by 'Group' and find the minimum Chi-Squared P-value
max_chi2_pvalues = all_results.groupby('Group')['Chi-Squared P-value'].max().reset_index()
# Merge to get all distributions with the minimum Chi-Squared P-value for each group
highest_chi2_pvalues  = pd.merge(all_results, max_chi2_pvalues, on=['Group', 'Chi-Squared P-value'])

#df to store all parameters
params = pd.DataFrame()
low_ks_values =[]
low_chi_values = []
#since the chi p value is always above 1 when we are maxing on ks values we are looking at the highest ks
for index, row in highest_ks_pvalues.iterrows():
    if row['KS P-value'] > 0.05:
        group_data = data[data['Group'] == row['Group']]['Duration_Surgery']
        row_parameters = fit_distribution(group_data, row['distribution'])
        row = pd.DataFrame({
            'Group': [row['Group']],
            'Distribution_ward_stay': [row['distribution']],
            'Shape_ward_stay': [row_parameters[0]],
            'Scale_ward_stay': [row_parameters[2]]})
        params = pd.concat([params, row], ignore_index=True)
    else:
        low_ks_values.append(row['Group'])

def assign_distribution(data, group_number, distribution):
    #En dan handmatig voor de laatste 5, op basis van de QQ plots
    group_data = data[data['Group'] == group_number]['Duration_Surgery']
    row_parameters = fit_distribution(group_data, distribution)
    row = pd.DataFrame({
        'Group': [group_number],
        'Distribution_ward_stay': [distribution],
        'Shape_ward_stay': [row_parameters[0]],
        'Scale_ward_stay': [row_parameters[2]]})
    return row
weibull_list = [6,7,10,16,20,22, 23,34,35,40,49,50]
for i in weibull_list:
    current_row = assign_distribution(data, i, 'weibull')
    params = pd.concat([params, current_row], ignore_index=True)

current_row = assign_distribution(data, 9, 'log-logistic')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 25, 'gamma')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 26, 'gamma')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 33, 'lognorm')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 48, 'lognorm')
params = pd.concat([params, current_row], ignore_index=True)

#create a column with averagere surgery duration per group to the parameter dataframe
average_duration_per_group = data.groupby('Group')['Total_Time_Ward'].mean().reset_index()

#create a dataframe to store the parameters, averages and group and category number
group_data = data.groupby('Group').agg({'Category': 'first', 'Group number': 'first'}).reset_index()
group_data = pd.merge(group_data, average_duration_per_group)
params = pd.merge(params, group_data, on='Group', how='left')

params = params.drop(columns = ['Group number', 'Category'])
params = pd.merge(distribution_parameters_surgery_duration, params, on='Group', how='left')
params.to_excel('Model_data/Distributie_parameters.xlsx')
all_results.to_excel('Ward_stay_test_results.xlsx')

insignificat_groups = [6, 7, 9, 10, 16, 20, 22, 23, 25, 26, 33, 34, 35, 40, 48, 49, 50]
fig, axes = plt.subplots(nrows=len(insignificat_groups), ncols=5, figsize=(30, 3*len(insignificat_groups)))  # Large figure size to accommodate all subplots

# Iterate over each group to perform analyses and plotting
for group_index, group in enumerate(insignificat_groups):
    group_data = data[data['Group'] == group]['Total_Time_Ward']
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    # Dictionary to store results for this category
    results = {}

    for i, dist in enumerate(distributions):
        # Fit distribution and perform tests
        params = fit_distribution(group_data, dist)
        # Generate QQ plot in the assigned subplot
        ax = axes[group_index, i]  # Select the appropriate subplot
        plot_qq(group_data, dist, params, ax)
        ax.set_title(f'{dist.capitalize()} (Group: {group})')

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('qq_plots_ward_stay_insignificant_groups.png')