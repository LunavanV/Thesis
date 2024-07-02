import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import StringIO

import numpy as np

# Set pandas display options for better output readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# Load the data
data_path = 'Model_data/With_outliers_removed.xlsx'  # Adjust the path as needed
data = pd.read_excel(data_path)

data = data.rename(columns={'Total_Time_Ward':'Ward1_time'})
# Extract ward number from the column name
data['Total_Time_Ward'] = data['Ward1_time'].fillna(0) + data['Ward2_time'].fillna(0) + data['Ward3_time'].fillna(0) + data['Ward4_time'].fillna(0)

print(len(data))

data = data.dropna(thresh=10, axis=1) #deleting the columns with less then 10 entries


data['Group'] = data.groupby(['Category', 'Group number']).ngroup() + 1

# Calculating the 10th and 90th percentiles of 'Total_Time_Ward' for each 'Group'
percentiles = data.groupby('Group')['Total_Time_Ward'].quantile([0.1, 0.9]).unstack()
percentiles.columns = ['Percentile10', 'Percentile90']

# Merging the percentile data back to the original dataframe
data = data.merge(percentiles, on='Group', how='left')

# Filtering out rows where 'Total_Time_Ward' is outside the 10th to 90th percentile range
data = data[(data['Total_Time_Ward'] >= data['Percentile10']) & (data['Total_Time_Ward'] <= data['Percentile90'])]

data = data[~((data['Group'] == 28) & (data['Total_Time_Ward'] > 20000))]

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot of Ward_Time
axes[0].boxplot(data['Total_Time_Ward'])
axes[0].set_title('Boxplot of Ward_Time')
axes[0].set_ylabel('Ward_Time')

# Histogram of Ward_Time
axes[1].hist(data['Total_Time_Ward'], bins=30, edgecolor='black')
axes[1].set_title('Histogram of Ward_Time')
axes[1].set_xlabel('Ward_Time')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Duration_Surgery'], data['Total_Time_Ward'], alpha=0.5)
plt.title('Scatter Plot of Duration_Surgery vs. Ward_Time')
plt.xlabel('Duration_Surgery')
plt.ylabel('Ward_Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['Group'], data['Total_Time_Ward'], alpha=0.5)
plt.title('Scatter Plot of Group vs. Ward_Time')
plt.xlabel('Group')
plt.ylabel('Ward_Time')
plt.grid(True)
plt.show()


# Create dummy variables for 'Category'
dummies = pd.get_dummies(data['Group'], prefix='Group')
dummies.drop('Group_50', axis=1, inplace=True)
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
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

results_html = model.summary().tables[1].as_html()
results_df = pd.read_html(StringIO(results_html), header=0, index_col=0)[0]
results_df.to_excel('Model_data/regression_summary.xlsx')

