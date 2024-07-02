import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, gamma, weibull_min, norm, fisk, invgamma, chisquare, kstest  # fisk is log-logistic in SciPy
#setting a randomly chosen seed in order to create reproducability
np.random.seed(50)

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

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

# Read the data
data = pd.read_excel('Input_Data/df_total.xlsx')
distribution_parameters = pd.read_excel('Model_data/Distributie_parameters.xlsx')
new_data = distribution_parameters[['Group', 'Category', 'Group number']]
data = data.merge(new_data, how='left', on=['Group number', 'Category'])

# List of groups to subset
groups_to_subset = [6, 16, 22, 33, 34, 48, 49]

# Filter the data to include only the specified groups
data = data[data['Group'].isin(groups_to_subset)]
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

fig, axes = plt.subplots(nrows=len(groups), ncols=3, figsize=(18, 6 * len(groups)))

for i, group in enumerate(groups):
    group_data = data[data['Group'] == group].dropna(subset=['Total_Time_Ward', 'Duration_Surgery'])

    # Histogram on the left
    sns.histplot(group_data['Total_Time_Ward'], kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Histogram for Group {group}')

    # Boxplot on the right
    sns.boxplot(x=group_data['Total_Time_Ward'], ax=axes[i, 1], color='blue')
    axes[i, 1].set_title(f'Boxplot for Group {group}')

    # Scatterplot
    sns.scatterplot(data=group_data, x='Duration_Surgery', y='Total_Time_Ward', ax=axes[i, 2], color='blue')
    axes[i, 2].set_title(f'Scatterplot for Group {group}')
    correlation = group_data['Duration_Surgery'].corr(group_data['Total_Time_Ward'])
    print(f"Correlation coefficient for group {group}:", correlation)

# Adjust layout
fig.tight_layout()
fig.savefig('group_regression_plots.png')

# Create dummy variables for 'Category'
dummies = pd.get_dummies(data['Group'], prefix='Group', drop_first= True) #
# dummies.drop('Group_42', axis=1, inplace=True) #setting the reference category
dummies = dummies.astype(int)

# Concatenate dummies with the original DataFrame
df = pd.concat([data, dummies], axis=1)

# Define independent variables (including dummy variables and 'Duration_Surgery')
X = df[dummies.columns.tolist()+ ['Duration_Surgery']]

# Add constant for intercept
X = sm.add_constant(X)

# Define dependent variable
y = df['Total_Time_Ward']

# Fit the regression model
Ward_model = sm.OLS(y, X).fit()
# Print the regression results
print(Ward_model.summary())
print(f'The r-squared is {round(Ward_model.rsquared,4)}')
# Get p-values
p_values = Ward_model.pvalues

# Count significant p-values
significant_count = (p_values > 0.05).sum()

print("Number of significant p-values:", significant_count)