import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing all functions needed to run the code
from Functions_distributions import fit_distribution, plot_qq, chi_squared_test, ks_test

# Set figure display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)
sns.set(style="whitegrid")

# Read the data
df = pd.read_excel('..//Input_Data/data_with_group_number.xlsx')

#remove negative numbers
df = df[df['Wardtime'] >= 0]
df = df[df['Duration_Surgery'] >= 0]

#split the data in test and train data
df_test = df[df['train-test'] == 'test'].copy()
df_train = df[df['train-test'] == 'train'].copy()
print(len(df_train))
print(len(df_test))

#Scatterplot of ward stay and surgery duration train data
sns.scatterplot(data=df_train, x='Duration_Surgery', y='Wardtime')
plt.title('Scatterplot of Duration of Surgery vs length of stay train data')
plt.xlabel('Duration of Surgery')
plt.ylabel('Wardtime')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('Total_data_set_correlation_train.png')  # Save the combined plot if needed


#Scatterplot of ward stay and surgery duration test data
sns.scatterplot(data=df_test, x='Duration_Surgery', y='Wardtime')
plt.title('Scatterplot of Duration of Surgery vs length of stay test data')
plt.xlabel('Duration of Surgery')
plt.ylabel('Wardtime')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('Total_data_set_correlation_test.png')  # Save the combined plot if needed

def set_boxplot_and_histogram(data, train_test, variable):
    """
    This function plots a histogram and a boxplot side by side for the specified variable in a given dataset.
    It saves the combined plot as a PNG file named 'Total_data_set_{variable}_{train_test}.png'.

    Parameters:
    - data (DataFrame): The input DataFrame containing the dataset to be plotted.
    - train_test (str): A label indicating the type of data (e.g., 'train', 'test', 'validation').
                        This is used for naming the saved plot file.
    - variable (str):   The name of the variable/column in the dataset that needs to be plotted.

    Returns/output:
    A boxplot and a hitogram
    """
    plt.figure(figsize=(12, 6))

    # Plotting histogram
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.histplot(data[variable], kde=True, bins='auto', color='blue')
    plt.title(f'Histogram of {variable} - {train_test} data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plotting boxplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.boxplot(y=data[variable], color='lightblue')
    plt.title(f'Boxplot of {variable} - {train_test} data')
    plt.ylabel('Value')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'Total_data_set_{variable}_{train_test}.png')  # Save the combined plot

#creating a boxplot and histogram for all the data and improt variabels
set_boxplot_and_histogram(df_test, 'test', 'Wardtime')
set_boxplot_and_histogram(df_train, 'train', 'Wardtime')
set_boxplot_and_histogram(df_test, 'test', 'Duration_Surgery')
set_boxplot_and_histogram(df_train, 'train', 'Duration_Surgery')
set_boxplot_and_histogram(df, 'total', 'Wardtime')
set_boxplot_and_histogram(df, 'total', 'Duration_Surgery')


# Define the distributions to fit
distributions = ['gamma', 'lognorm', 'weibull', 'log-logistic', 'pearsonV']

# Fit each distribution to the train data
params_wardtime = {}
params_surgery_duation = {}
for dist in distributions:
    params_wardtime[dist] = fit_distribution(df_train['Wardtime'], dist)
    params_surgery_duation[dist] = fit_distribution(df_train['Duration_Surgery'], dist)

# Plot QQ plots for each distribution
fig, axes = plt.subplots(nrows=2, ncols=len(distributions), figsize=(20, 10))

for i, dist in enumerate(distributions):
    plot_qq(df_train['Wardtime'], dist, params_wardtime[dist], axes[0, i])
    axes[0, i].set_ylim(0, 500000)  # Set y-axis limit for the first row
    axes[0, i].set_xlim(0, 500000)  # Set x-axis limit for the first row
    plot_qq(df_train['Duration_Surgery'], dist, params_surgery_duation[dist], axes[1, i])

#making sure all the axes are alined
axes[1, 0].set_xlim(0, 1200)  # Set x-axis limit for the second row 1th plot
axes[1, 0].set_ylim(0, 1200)  # Set y-axis limit for the second row 1th plot
axes[1, 1].set_xlim(0, 1200)  # Set x-axis limit for the second row 2nd plot
axes[1, 1].set_ylim(0, 1200)  # Set y-axis limit for the second row 2nd plot
axes[1, 2].set_xlim(0, 1200)  # Set x-axis limit for the second row 3th plot
axes[1, 2].set_ylim(0, 1200)  # Set y-axis limit for the second row 3th plot
axes[1, 3].set_xlim(0, 1200)  # Set x-axis limit for the second row 4th plot
axes[1, 3].set_ylim(0, 1200)  # Set x-axis limit for the second row 4th plot
axes[1, 4].set_ylim(0, 1200)  # Set y-axis limit for the second row 5th plot
axes[1, 4].set_xlim(0, 1200)  # Set x-axis limit for the second row 5th plot
# Set titles for each row separately
axes[0, 0].set_ylabel('Sample Quantiles (Lenght of Stay)')
axes[1, 0].set_ylabel('Sample Quantiles (Duration Surgery)')
plt.tight_layout()
plt.savefig('Total_data_qq_plots_train.png')  # Save the combined plot if needed

#testing all the training data lenght of stay for their fit
chi_squared_results_wardtime = {}
ks_results_wardtime = {}

for dist in distributions:
    chi_squared_results_wardtime[dist] = chi_squared_test(df_train['Wardtime'], dist, params_wardtime[dist])
    ks_results_wardtime[dist] = ks_test(df_train['Wardtime'], dist, params_wardtime[dist])

# Prepare the results for display
chi_squared_ks_results_df_wardtime = pd.DataFrame({
    'Distribution': distributions,
    'Chi-Squared Statistic': [chi_squared_results_wardtime[dist][0] for dist in distributions],
    'Chi-Squared p-value': [chi_squared_results_wardtime[dist][1] for dist in distributions],
    'KS Statistic': [ks_results_wardtime[dist].statistic for dist in distributions],
    'KS p-value': [ks_results_wardtime[dist].pvalue for dist in distributions]
})
print('fitting results length of stay')
print(chi_squared_ks_results_df_wardtime)

#testing all the training data Surgery duration for their fit
chi_squared_results_Duration_Surgery = {}
ks_results_Duration_Surgery = {}

for dist in distributions:
    chi_squared_results_Duration_Surgery[dist] = chi_squared_test(df_train['Duration_Surgery'], dist, params_surgery_duation[dist])
    ks_results_Duration_Surgery[dist] = ks_test(df_train['Duration_Surgery'], dist, params_surgery_duation[dist])

# Prepare the results for display
chi_squared_ks_results_df_Duration_Surgery = pd.DataFrame({
    'Distribution': distributions,
    'Chi-Squared Statistic': [chi_squared_results_Duration_Surgery[dist][0] for dist in distributions],
    'Chi-Squared p-value': [chi_squared_results_Duration_Surgery[dist][1] for dist in distributions],
    'KS Statistic': [ks_results_Duration_Surgery[dist].statistic for dist in distributions],
    'KS p-value': [ks_results_Duration_Surgery[dist].pvalue for dist in distributions]
})

print('fitting results surgery duration')
print(chi_squared_ks_results_df_Duration_Surgery)


# Group by 'Group', 'Category', and 'Group_number' and count occurrences in train and test
train_counts = df_train.groupby(['Group', 'Category', 'Group_number']).size().reset_index(name='Count_in_train')
test_counts = df_test.groupby(['Group', 'Category', 'Group_number']).size().reset_index(name='Count_in_test')

# Merge train and test counts based on 'Group', 'Category', and 'Group_number'
group_counts = pd.merge(train_counts, test_counts, on=['Group', 'Category', 'Group_number'], how='outer')

# Fill NaN values with 0 (in case some groups only appear in either train or test)
group_counts = group_counts.fillna(0)

print(group_counts)

# Define the bins
bins = [0, 500, 1000, 1500, 2000, 3000,4000,5000,10000,15000,20000,30000, float('inf')]  # Define your bins here

# Use pd.cut to categorize the data
bin_labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000-4000', '4000-5000', '5000-10000', '10000-15000','15000-20000', '20000-30000', '30000+']
df_train['Wardtime_bins'] = pd.cut(df_train['Wardtime'], bins=bins, labels=bin_labels, right=False)

# Count the occurrences in each bin
bin_counts = df_train['Wardtime_bins'].value_counts().sort_index()

# Print or use the counts as needed
print("Count of data points in each bin:")
print(bin_counts)