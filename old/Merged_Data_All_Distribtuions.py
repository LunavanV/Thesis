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
data = pd.read_excel('Input_Data/Clean_data_OR+Ward.xlsx')

groups = data['Group'].unique()
groups = sorted(groups)
# Setting the style for the plots
sns.set(style="whitegrid")

# data = data[data['ORtime'] <= 750]
data = data[data['ORtime'] >= 0]

print(len(data))

sns.histplot(data['ORtime'], kde=True, bins='auto', color='blue')
plt.title('Histogram of Surgery Durations')
plt.xlabel('Duration in Minutes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(y=data['ORtime'], color='lightblue')
plt.title('Boxplot of Surgery Durations')
plt.ylabel('Duration in Minutes')
plt.savefig('boxplot_surgery.png')
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
    try:
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
    except Exception as e:
        ax.text(0.5, 0.5, 'Fit Failed', ha='center', va='center', transform=ax.transAxes)
        print(f"Error plotting QQ plot for {dist} distribution with {group}: {str(e)}")
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
def remove_outliers(group, df, lowerbound = 0.25, upperbound = 0.75):
    if lowerbound == 0.0:
        Q1 = group['ORtime'].min()
    else:
        Q1 = group['ORtime'].quantile(lowerbound)
    Q3 = group['ORtime'].quantile(upperbound)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    group = group[(group['ORtime'] >= lower_bound) & (group['ORtime'] <= upper_bound)]
    df = pd.concat([df, group])

    # Filter and return the group without outliers
    return df

def testing_all(data, lowerbound, upperbound):
    # Iterate over each group
    grouped_data = data.groupby('Group')
    data_filtered = pd.DataFrame()
    # Iterate over each group
    for group_name, group_data in grouped_data:
        data_filtered = remove_outliers(group_data, data_filtered, lowerbound, upperbound)
    data = data_filtered
    data.sort_values(by=['Group'], ascending = True)
    print(len(data))
    # Assuming 'data' and 'groups' are predefined
    fig, axes = plt.subplots(nrows=50, ncols=6, figsize=(30, 150))  # Large figure size to accommodate all subplots
    distributions = ['gamma', 'log-logistic', 'lognorm', 'weibull', 'norm', 'pearsonV']
    all_results = pd.DataFrame()

    # Iterate over each group to perform analyses and plotting
    for group_index, group in enumerate(groups):
        group_data = data[data['Group'] == group]['ORtime']
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
    plt.savefig('qq_plots.png')
    plt.show()

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
    print(upperbound, lowerbound)
    # Print the overview
    print(overview)

low_list = [0.05, 0.1, 0.15,0.2]
high_list = [0.95,0.9,0.85,0.8]
for i in low_list:
    for j in high_list:
        testing_all(data, i,j)

