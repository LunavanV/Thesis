import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#importing all functions needed to run the code
from Functions_distributions import fit_distribution, plot_qq, chi_squared_test, ks_test

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read the data
df = pd.read_excel('../Input_Data/data_with_group_number.xlsx')
df = df[df['Duration_Surgery'] >= 0]

# Extract Duration_Surgery column
surgery_durations = df['Duration_Surgery']


# Define the distributions to fit
distributions = ['gamma', 'lognorm', 'weibull', 'log-logistic', 'pearsonV']

# Fit each distribution to the data
params = {}
for dist in distributions:
    params[dist] = fit_distribution(surgery_durations, dist)

# Plot QQ plots for each distribution
fig, axes = plt.subplots(nrows=1, ncols=len(distributions), figsize=(20, 5))

for ax, dist in zip(axes, distributions):
    plot_qq(surgery_durations, dist, params[dist], ax)

plt.tight_layout()
plt.show()

chi_squared_results = {}
ks_results = {}
ad_results = {}
print(params)

for dist in distributions:
    chi_squared_results[dist] = chi_squared_test(surgery_durations, dist, params[dist])
    ks_results[dist] = ks_test(surgery_durations, dist, params[dist])

# Prepare the results for display
chi_squared_ks_results_df = pd.DataFrame({
    'Distribution': distributions,
    'Chi-Squared Statistic': [chi_squared_results[dist][0] for dist in distributions],
    'Chi-Squared p-value': [chi_squared_results[dist][1] for dist in distributions],
    'KS Statistic': [ks_results[dist].statistic for dist in distributions],
    'KS p-value': [ks_results[dist].pvalue for dist in distributions]
})

print(chi_squared_ks_results_df)

expected = np.random.lognormal(mean=np.log(params['lognorm'][2]), sigma=params['lognorm'][0], size=18000)
sns.histplot(expected, kde=True, bins='auto', color='blue')
plt.title('Histogram of total length of stay')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()