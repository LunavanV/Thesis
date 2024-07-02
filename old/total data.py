import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, gamma, weibull_min, norm, fisk, invgamma, chisquare, kstest  # fisk is log-logistic in SciPy

# Set pandas display options
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_excel('Data/df_total.xlsx')
data['Group'] = data.groupby(['Category', 'Group number']).ngroup() + 1
# Setting the style for the plots
sns.set(style="whitegrid")

# Calculate Q1, Q3, and IQR
Q1 = data['Duration_Surgery'].quantile(0.25)
Q3 = data['Duration_Surgery'].quantile(0.75)
Q3 = data['Duration_Surgery'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
data = data[(data['Duration_Surgery'] >= lower_bound) & (data['Duration_Surgery'] <= upper_bound)]
data = data[(data['Duration_Surgery'] <= 180)]

print(len(data))

sns.histplot(data['Duration_Surgery'], kde=True, bins='auto', color='blue')
plt.title('Histogram of Surgery Durations')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Duration_Surgery'], color='lightblue')
plt.title('Boxplot of Surgery Durations')
plt.ylabel('Duration in Minutes')
plt.savefig('Data/boxplot_surgery.png')
plt.show()


# Helper function to fit distribution parameters
def fit_distribution(data, dist):
    if dist == 'gamma':
        return gamma.fit(data, floc=0)  # fixing location to 0
    elif dist == 'lognorm':
        return lognorm.fit(data, floc=0)  # fixing location to 0
    elif dist == 'weibull':
        return weibull_min.fit(data, floc=0)  # fixing location to 0
    elif dist == 'norm':
        return norm.fit(data)
    elif dist == 'log-logistic':
        return fisk.fit(data, floc=0)  # fisk is used for log-logistic
    elif dist == 'pearsonV':
        return invgamma.fit(data, floc=0)  # invgamma is Pearson Type V

# QQ plots for each distribution
def plot_qq(data, dist, params, ax):
    if dist == 'gamma':
        sm.qqplot(data, gamma, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    elif dist == 'lognorm':
        sm.qqplot(data, lognorm, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    elif dist == 'weibull':
        sm.qqplot(data, weibull_min, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    elif dist == 'norm':
        sm.qqplot(data, norm, loc=params[0], scale=params[1], line='45', ax=ax)
    elif dist == 'log-logistic':
        sm.qqplot(data, fisk, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    elif dist == 'pearsonV':
        sm.qqplot(data, invgamma, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    ax.set_title(f'QQ Plot: {dist.capitalize()}')


# Chi-squared test for goodness of fit
def chi_squared_test(data, dist, params):
    """Perform chi-squared test to compare observed data with a fitted distribution."""
    observed, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if dist == 'gamma':
        expected = gamma.pdf(bin_centers, *params)
    elif dist == 'lognorm':
        expected = lognorm.pdf(bin_centers, *params)
    elif dist == 'weibull':
        expected = weibull_min.pdf(bin_centers, *params)
    elif dist == 'norm':
        expected = norm.pdf(bin_centers, *params)
    elif dist == 'log-logistic':
        expected = fisk.pdf(bin_centers, *params)  # fisk is used for log-logistic
    elif dist == 'pearsonV':
        expected = invgamma.pdf(bin_centers, *params)  # invgamma is Pearson Type V
    expected = expected / expected.sum() * observed.sum()  # Normalize to match observed counts
    return chisquare(observed, expected)

# Kolmogorov-Smirnov test for each distribution
def ks_test(data, dist, params):
    """Perform KS test to compare empirical data with a theoretical distribution."""
    if dist == 'gamma':
        return kstest(data, 'gamma', args=params)
    elif dist == 'lognorm':
        return kstest(data, 'lognorm', args=params)
    elif dist == 'weibull':
        return kstest(data, 'weibull_min', args=params)
    elif dist == 'norm':
        return kstest(data, 'norm', args=params)
    elif dist == 'log-logistic':
        return kstest(data, 'fisk', args=params)  # fisk is used for log-logistic
    elif dist == 'pearsonV':
        return kstest(data, 'invgamma', args=params)  # invgamma for Pearson Type V


# Main analysis
data = data[data['Duration_Surgery'] >= 0]
durations = data['Duration_Surgery'].dropna()  # Ensure no NaN values
results = {}
# Create a figure with multiple subplots for QQ plots
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))

distributions = ['gamma', 'lognorm', 'weibull', 'norm', 'log-logistic', 'pearsonV']

for i, dist in enumerate(distributions):    # Fit distribution and generate plots and tests
    params = fit_distribution(durations, dist)
    plot_qq(durations, dist, params, axes[i])
    chi2_result = chi_squared_test(durations, dist, params)
    ks_result = ks_test(durations, dist, params)
    results[dist] = {
        'Parameters': params,
        'Chi-Squared Statistic': chi2_result.statistic,
        'Chi-Squared P-value': chi2_result.pvalue,
        'KS Statistic': ks_result.statistic,
        'KS P-value': ks_result.pvalue
    }

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()

results_df = pd.DataFrame(results).T  # Transpose to make distributions the rows
print(results_df)