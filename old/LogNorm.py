import pandas as pd
import numpy as np
from scipy.stats import lognorm, kstest, chisquare  # Import lognorm distribution and statistical tests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Load the data
data = pd.read_excel('Input_Data/df_total.xlsx')

# Add a unique group identifier if needed
data['Group'] = data.groupby(['Category', 'Group number']).ngroup() + 1

# Prepare to store the results
lognorm_params = pd.DataFrame(columns=['Group', 'Shape', 'Location', 'Scale'])  # DataFrame for parameters
ks_results = pd.DataFrame(columns=['Group', 'KS Statistic', 'KS P-Value'])  # DataFrame for KS test results
chi_results = pd.DataFrame(columns=['Group', 'Chi Statistic', 'Chi P-Value'])  # DataFrame for Chi-square test results

groups = data['Group'].unique()

data_filtered = pd.DataFrame()

# Function to remove outliers within each group
def remove_outliers(group, df):
    Q1 = group['Duration_Surgery'].quantile(0.25)
    Q3 = group['Duration_Surgery'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group = group[(group['Duration_Surgery'] >= lower_bound) & (group['Duration_Surgery'] <= upper_bound)]
    df = pd.concat([df, group])

    # Filter and return the group without outliers
    return df

grouped_data = data.groupby('Group')

# Iterate over each group
for group_name, group_data in grouped_data:
    data_filtered = remove_outliers(group_data, data_filtered)
data = data_filtered

fig, axes = plt.subplots(nrows=len(groups), ncols=2, figsize=(15, 6 * len(groups)))

for i, group in enumerate(groups):
    group_data = data[data['Group'] == group]['Duration_Surgery'].dropna()
    group_data = group_data[group_data >= 0].dropna()  # Clean data

    # Histogram on the left
    sns.histplot(group_data, kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Histogram for Group {group}')

    # Boxplot on the right
    sns.boxplot(x=group_data, ax=axes[i, 1], color='blue')
    axes[i, 1].set_title(f'Boxplot for Group {group}')

    # Fit lognormal distribution
    params = lognorm.fit(group_data)
    row = pd.DataFrame({'Group': [group], 'Shape': [params[0]], 'Location': [params[1]], 'Scale': [params[2]]})
    lognorm_params = pd.concat([lognorm_params, row])

    # Perform and store KS test
    ks_stat, ks_p_value = kstest(group_data, 'lognorm', args=(params[0], params[1], params[2]))
    row = pd.DataFrame({'Group': [group], 'KS Statistic': [ks_stat], 'KS P-Value': [ks_p_value]})
    ks_results = pd.concat([ks_results, row])

    # Perform and store Chi-squared test
    observed, bin_edges = np.histogram(group_data, bins='auto', density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Midpoints of bins
    expected = lognorm.pdf(bin_centers, *params)  # Using the params from lognorm fitting
    # Scaling the expected frequencies
    expected_scaled = expected * (observed.sum() / expected.sum())
    # Now use the scaled expected frequencies for the Chi-squared test
    chi_stat, chi_p_value = chisquare(observed, expected_scaled)
    row = pd.DataFrame({'Group': [group], 'Chi Statistic': [chi_stat], 'Chi P-Value': [chi_p_value]})
    chi_results = pd.concat([chi_results, row])

# Adjust layout
fig.tight_layout()
plt.show()

# Merge all results into one DataFrame
test_results = pd.merge(ks_results, chi_results, on='Group')

average_duration_per_group = data.groupby('Group')['Duration_Surgery'].mean().reset_index()

group_data = data.groupby('Group').agg({'Category': 'first', 'Group number': 'first'}).reset_index()
group_data = pd.merge(group_data, average_duration_per_group)
lognorm_params = pd.merge(lognorm_params, group_data, on='Group', how='left')
print(lognorm_params)
print(test_results)
