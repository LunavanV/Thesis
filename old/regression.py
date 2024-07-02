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


# Suppress all future warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Load the data
data = pd.read_excel('Input_Data/df_total.xlsx')

# Add a unique group identifier if needed
data['Group'] = data.groupby(['Category', 'Group number']).ngroup() + 1

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

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
grouped_data = data.groupby('Group')
#create a new dataframe to store after removing outliers
data_filtered =pd.DataFrame()
# Iterate over each group and remove outliers
for group_name, group_data in grouped_data:
    data_filtered = remove_outliers(group_data, data_filtered, 'Duration_Surgery', 0.1, 0.9)
data = data_filtered
#setup a figure to store qq plots
#dit stukje moet eigenlijk al bij de data creatie
data = data.rename(columns={'Total_Time_Ward':'Ward1_time'})

# Extract ward number from the column name
data['Total_Time_Ward'] = data['Ward1_time'].fillna(0) + data['Ward2_time'].fillna(0) + data['Ward3_time'].fillna(0) + data['Ward4_time'].fillna(0)
data = data.dropna(thresh=10, axis=1) #deleting the columns with less then 10 entries

#create a new dataframe to store after removing outliers
data_filtered =pd.DataFrame()
grouped_data = data.groupby('Group')
# Iterate over each group and remove outliers
for group_name, group_data in grouped_data:
    data_filtered = remove_outliers(group_data, data_filtered, 'Total_Time_Ward', 0.15,0.85)
data = data_filtered
# data = data[~((data['Group'] == 23) & (data['Total_Time_Ward'] > 3100))]

print(len(data))
# setup a figure to store qq plots
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

results_html = Ward_model.summary().tables[1].as_html()
df_regression_results = pd.read_html(StringIO(results_html), header=0, index_col=0)[0]

average_length_of_stay = data.groupby('Group')['Total_Time_Ward'].mean().reset_index()
average_length_of_stay = average_length_of_stay.rename(columns={'Total_Time_Ward':'Average_Lenght_Of_Stay'})

ward_columns = ['Ward', 'Ward2', 'Ward3', 'Ward4']
ic_keywords = ['IC']
data['Assigned_to_IC'] = data[ward_columns].apply(lambda row: any(ic in str(w) for w in row for ic in ic_keywords), axis=1)

# Calculate the percentage of IC visits per category
percentage_ic_visits = (data.groupby('Group')['Assigned_to_IC'].sum() / data.groupby('Group')['Assigned_to_IC'].count()) * 100
IC_Visits_average_length_of_stay = pd.merge(average_length_of_stay, percentage_ic_visits, on='Group', how='left')
#Optionally, save results to Excel
IC_Visits_average_length_of_stay.to_excel('Model_data/IC_Visits_average_length_of_stay.xlsx')
df_regression_results.to_excel('Model_data/regression_summary.xlsx')
data.to_excel('Model_data/With_outliers_removed.xlsx')
