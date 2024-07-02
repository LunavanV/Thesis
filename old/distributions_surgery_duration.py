import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#setting a randomly chosen seed in order to create reproducability
np.random.seed(50)

#importing all functions needed to run the code
from Functions_distributions import fit_distribution, plot_qq, chi_squared_test, ks_test, remove_outliers

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_excel('Input_Data/df_total.xlsx')
data['Group'] = data.groupby(['Category', 'Group number']).ngroup() + 1

groups = data['Group'].unique()
groups = sorted(groups)
# Setting the style for the plots
sns.set(style="whitegrid")

# data = data[data['Duration_Surgery'] <= 750]
data = data[data['Duration_Surgery'] >= 0]

sns.histplot(data['Duration_Surgery'], kde=True, bins='auto', color='blue')
plt.title('Histogram of Surgery Durations')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Duration_Surgery'], color='lightblue')
plt.title('Boxplot of Surgery Durations')
plt.ylabel('Duration in Minutes')
plt.savefig('boxplot_surgery.png')
plt.show()

# Iterate over each group and remove the outliers
grouped_data = data.groupby('Group')
data_filtered = pd.DataFrame()
for group_name, group_data in grouped_data:
    filtered_group = remove_outliers(group_data, 'Duration_Surgery', 0.1, 0.9)
    data_filtered = pd.concat([data_filtered, filtered_group], ignore_index=True)
data = data_filtered

# Setting up the distributions that need to be fit and the figure in which to fit them
fig, axes = plt.subplots(nrows=50, ncols=5, figsize=(30, 150))  # Large figure size to accommodate all subplots
distributions = ['gamma', 'weibull', 'lognorm','log-logistic',  'pearsonV']

all_results = pd.DataFrame()

# Iterate over each group to perform analyses and plotting
for group_index, group in enumerate(groups):
    group_data = data[data['Group'] == group]['Duration_Surgery']
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
        #merge results with total dataframe
        all_results = pd.concat([all_results, row])

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('qq_plots_surgery_duration.png')

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
#since the chi p value is always above 1 when we are maxing on ks values we are looking at the highest ks
for index, row in highest_ks_pvalues.iterrows():
    if row['KS P-value'] > 0.05:
        group_data = data[data['Group'] == row['Group']]['Duration_Surgery']
        row_parameters = fit_distribution(group_data, row['distribution'])
        row = pd.DataFrame({
            'Group': [row['Group']],
            'Distribution_surgery_duration': [row['distribution']],
            'Shape_surgery_duration': [row_parameters[0]],
            'Scale_surgery_duration': [row_parameters[2]]})
        params = pd.concat([params, row], ignore_index=True)
    else:
        low_ks_values.append(row['Group'])

def assign_distribution(data, group_number, distribution):
    #En dan handmatig voor de laatste 5, op basis van de QQ plots
    group_data = data[data['Group'] == group_number]['Duration_Surgery']
    row_parameters = fit_distribution(group_data, distribution)
    row = pd.DataFrame({
        'Group': [group_number],
        'Distribution_surgery_duration': [distribution],
        'Shape_surgery_duration': [row_parameters[0]],
        'Scale_surgery_duration': [row_parameters[2]]})
    return row

current_row = assign_distribution(data, 22, 'weibull')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 43, 'weibull')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 44, 'lognorm')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 49, 'lognorm')
params = pd.concat([params, current_row], ignore_index=True)


#create a column with averagere surgery duration per group to the parameter dataframe
average_duration_per_group = data.groupby('Group')['Duration_Surgery'].mean().reset_index()

#create a dataframe to store the parameters, averages and group and category number
group_data = data.groupby('Group').agg({'Category': 'first', 'Group number': 'first'}).reset_index()
group_data = pd.merge(group_data, average_duration_per_group)
params = pd.merge(params, group_data, on='Group', how='left')

params.to_excel('Model_data/Distributie_parameters_only_surgery_duration.xlsx')
all_results.to_excel('Surgery_Duration_test_results.xlsx')




