import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting a randomly chosen seed to ensure reproducibility
np.random.seed(50)

# Importing all necessary functions
from Functions_distributions import fit_distribution, plot_qq, chi_squared_test, ks_test

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_excel('../Input_Data/data_with_group_number.xlsx')

# Get unique groups and sort them
groups = data['Group'].unique()
groups = sorted(groups)

# Setting the style for the plots
sns.set(style="whitegrid")

# Remove negative numbers
data = data[data['Wardtime'] >= 0]
data = data[data['Duration_Surgery'] >= 0]

# Split the data into test and train data
df_test = data[data['train-test'] == 'test'].copy()
df_train = data[data['train-test'] == 'train'].copy()

# List of distributions to fit
distributions = ['gamma', 'weibull', 'lognorm', 'log-logistic', 'pearsonV']

# Plotting histograms and boxplots for each group
fig, axes = plt.subplots(nrows=len(groups), ncols=2,
                         figsize=(20, 5 * len(groups)))  # Large figure size to accommodate all subplots

for group_index, group in enumerate(groups):
    group_data = df_train[df_train['Group'] == group]['Duration_Surgery']
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    # Generate histogram using sns.histplot
    ax_hist = axes[group_index, 0]  # Select the appropriate subplot for histogram
    sns.histplot(group_data, kde=True, bins='auto', ax=ax_hist, color='blue')
    ax_hist.set_title(f'Histogram (Group: {group})')
    ax_hist.set_xlabel('Duration of Surgery')
    ax_hist.set_ylabel('Frequency')

    # Generate boxplot
    ax_box = axes[group_index, 1]  # Select the appropriate subplot for boxplot
    sns.boxplot(x=group_data, ax=ax_box, color='lightblue')
    ax_box.set_title(f'Boxplot (Group: {group})')
    ax_box.set_xlabel('Duration of Surgery')
    ax_box.set_yticks([])  # Hide y-axis ticks

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('../Input variable creation/histogram_boxplot_surgery_duration.png')
plt.close()  # Close the figure after saving to release resources

# Fitting distributions and plotting QQ plots
fig, axes = plt.subplots(nrows=50, ncols=5, figsize=(30, 150))  # Large figure size to accommodate all subplots
all_results = pd.DataFrame()

# Iterate over each group to perform analyses and plotting
for group_index, group in enumerate(groups):
    group_data = df_train[df_train['Group'] == group]['Duration_Surgery']
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

        # Perform Chi-squared and KS tests
        chi2_result = chi_squared_test(group_data, dist, params)
        ks_result = ks_test(group_data, dist, params)

        # Store results in a row for the dataframe
        row = pd.DataFrame({
            'distribution': [dist],
            'Group': [group],
            'Parameters': [params],
            'Chi-Squared Statistic': [chi2_result[0]],
            'Chi-Squared P-value': [chi2_result[1]],
            'KS Statistic': [ks_result.statistic],
            'KS P-value': [ks_result.pvalue]
        })

        # Concatenate row to all_results dataframe
        all_results = pd.concat([all_results, row])

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('qq_plots_surgery_duration.png')
plt.close()  # Close the figure after saving to release resources

# Counting significant Chi-squared and KS tests
chi_counts = all_results.groupby('distribution').agg(
    ChiSquared_001=('Chi-Squared P-value', lambda x: (x < 0.01).sum()),
    ChiSquared_005=('Chi-Squared P-value', lambda x: (x < 0.05).sum())
).reset_index()

ks_counts = all_results.groupby('distribution').agg(
    KS_001=('KS P-value', lambda x: (x < 0.01).sum()),
    KS_005=('KS P-value', lambda x: (x < 0.05).sum())
).reset_index()

# Merging the counts
overview = pd.merge(chi_counts, ks_counts, on='distribution', how='outer')

# Fill missing values with 0
overview = overview.fillna(0)
print(overview)

# Identifying significant groupings based on significance level
significant_groups = all_results[(all_results['Chi-Squared P-value'] < 0.05) & (all_results['KS P-value'] < 0.05)][
    'Group'].unique()
significant_groups_df = all_results[(all_results['Chi-Squared P-value'] < 0.05) & (all_results['KS P-value'] < 0.05)]

# Extracting parameters for significant groups
params = pd.DataFrame()
for group in significant_groups_df['Group'].unique():
    group_data = significant_groups_df[significant_groups_df['Group'] == group]
    min_chi_statistic = group_data['Chi-Squared P-value'].min()
    min_chi_rows = group_data[group_data['Chi-Squared P-value'] == min_chi_statistic]
    min_chi_rows = min_chi_rows.sort_values(by=['KS P-value'], ascending=True)
    params = pd.concat([params, min_chi_rows.head(1)])

# Convert 'Parameters' column to string type explicitly
params['Parameters'] = params['Parameters'].astype(str)

# Split Parameters column into separate columns
params[['Shape_surgery_duration', 'Loc_surgery_duration', 'Scale_surgery_duration']] = params['Parameters'].str.strip(
    '()').str.split(', ', expand=True)

# Drop unnecessary columns and renaming to make it more clear when merging later
params = params.drop(columns=['Loc_surgery_duration', 'Parameters', 'Chi-Squared Statistic', 'Chi-Squared P-value', 'KS Statistic', 'KS P-value'])
params = params.rename(columns={'distribution': 'Distribution_surgery_duration'})

# Filter out insignificant groups
insignificant_groups_df = all_results[~all_results['Group'].isin(significant_groups_df['Group'].unique())]
insignificant_groups = insignificant_groups_df['Group'].unique()

# Plotting QQ plots for insignificant groups
fig, axes = plt.subplots(nrows=len(insignificant_groups), ncols=5,
                         figsize=(40, 5 * len(insignificant_groups)))  # Large figure size to accommodate all subplots

# Iterate over each group to perform analyses and plotting
for group_index, group in enumerate(insignificant_groups):
    group_data = df_train[df_train['Group'] == group]['Duration_Surgery']
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    for i, dist in enumerate(distributions):
        # Fit distribution and perform tests
        parameters = fit_distribution(group_data, dist)
        # Generate QQ plot in the assigned subplot
        ax = axes[group_index, i]  # Select the appropriate subplot
        plot_qq(group_data, dist, parameters, ax)
        ax.set_title(f'{dist.capitalize()} (Group: {group})')

plt.tight_layout()  # Adjust layout to make labels and titles readable
plt.savefig('qq_plots_surgery_duration_insignificant_groups.png')
plt.close()  # Close the figure after saving to release resources


# Assigning distributions manually based on QQ plots for specific groups
def assign_distribution(data, group_number, distribution):
    group_data = data[data['Group'] == group_number]['Duration_Surgery']
    row_parameters = fit_distribution(group_data, distribution)
    row = pd.DataFrame({
        'Group': [group_number],
        'Distribution_surgery_duration': [distribution],
        'Shape_surgery_duration': [row_parameters[0]],
        'Scale_surgery_duration': [row_parameters[2]]
    })
    return row


# List of specific groups to assign distributions
log_norm_list = [3, 4, 8, 10, 12, 30, 34, 48]

# Assign distributions for each group in log_norm_list
for i in log_norm_list:
    current_row = assign_distribution(data, i, 'lognorm')
    params = pd.concat([params, current_row], ignore_index=True)

# Assign distributions manually for other specific groups
current_row = assign_distribution(data, 7, 'gamma')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 11, 'gamma')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 31, 'gamma')
params = pd.concat([params, current_row], ignore_index=True)

current_row = assign_distribution(data, 27, 'log-logistic')
params = pd.concat([params, current_row], ignore_index=True)
current_row = assign_distribution(data, 50, 'log-logistic')
params = pd.concat([params, current_row], ignore_index=True)

current_row = assign_distribution(data, 28, 'weibull')
params = pd.concat([params, current_row], ignore_index=True)

current_row = assign_distribution(data, 45, 'pearsonV')
params = pd.concat([params, current_row], ignore_index=True)

# Calculate average surgery duration per group and add to params dataframe
average_duration_per_group = df_train.groupby('Group')['Duration_Surgery'].mean().reset_index()

# Create dataframe to store parameters, averages, group, and category number
group_data = data.groupby('Group').agg({'Category': 'first', 'Group_number': 'first'}).reset_index()
group_data = pd.merge(group_data, average_duration_per_group)
params = pd.merge(params, group_data, on='Group', how='left')
params = params.rename({'Duration_Surgery' : 'Avarage_Duration_Surgery'})

# Save results to Excel files
params.to_excel('../Input_Data/Distributie_parameters_surgery_duration.xlsx', index=False)
print(params)
all_results.to_excel('Surgery_Duration_test_results.xlsx')
