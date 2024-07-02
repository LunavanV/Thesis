import numpy as np
import statsmodels.api as sm
from scipy.stats import lognorm, gamma, weibull_min,fisk, invgamma, chisquare, kstest  # fisk is log-logistic in SciPy
#setting a randomly chosen seed in order to create reproducability
np.random.seed(50)

def fit_distribution(data, dist):
    """
       Fit a specified distribution to the given data and return the parameters.

       Parameters:
       - data (Pandas series): The data to fit the distribution to.
       - dist (str): The name of the distribution to fit ('gamma', 'lognorm', 'weibull', 'log-logistic', 'pearsonV').

       Returns:
       - params (tuple): Parameters of the fitted distribution.
    """

    if dist == 'gamma':
        return gamma.fit(data, floc=0)  # fixing location to 0
    elif dist == 'lognorm':
        return lognorm.fit(data, floc=0)  # fixing location to 0
    elif dist == 'weibull':
        return weibull_min.fit(data, floc=0)  # fixing location to 0
    elif dist == 'log-logistic':
        return fisk.fit(data, floc=0)  # fisk is used for log-logistic
    elif dist == 'pearsonV':
        return invgamma.fit(data, floc=0)  # invgamma is Pearson Type V

def plot_qq(data, dist, params, ax):
    """
       Plot a Q-Q plot for a specified distribution against the given data.

       Parameters:
       - data (Pandas series): The data to plot against the distribution.
       - dist (str): The name of the distribution ('gamma', 'lognorm', 'weibull', 'log-logistic', 'pearsonV').
       - params (tuple): Parameters of the fitted distribution.
       - ax (matplotlib.axes.Axes): The axis object to draw the plot on.

       Returns/Output:
       Creates the defined QQ plot for the chose distribution
    """
    try:
        # Plot Q-Q plot based on the specified distribution
        if dist == 'gamma':
            sm.qqplot(data, gamma, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
        elif dist == 'lognorm':
            sm.qqplot(data, lognorm, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
        elif dist == 'weibull':
            sm.qqplot(data, weibull_min, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
        elif dist == 'log-logistic':
            sm.qqplot(data, fisk, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
        elif dist == 'pearsonV':
            sm.qqplot(data, invgamma, distargs=(params[0],), loc=params[1], scale=params[2], line='45', ax=ax)
    except Exception as e:
        # Handle plot failure by displaying 'Fit Failed' and printing the error message
        ax.text(0.5, 0.5, 'Fit Failed', ha='center', va='center', transform=ax.transAxes)
        print(f"Error plotting QQ plot for {dist} distribution with {group}: {str(e)}")
    ax.set_title(f'QQ Plot: {dist.capitalize()}')

# Chi-squared test for goodness of fit
def chi_squared_test(data, dist, params, bins='auto', smoothing=0.01):
    """
        Create expected data of a fitted distribution so this can be compared to the observed data by performing a
        chi-squared test. Additionally the data is smoothened in order to avoid dividing by zero.

        Parameters:
        - data (pandas series): The observed data to be tested.
        - dist (str): The name of the distribution ('gamma', 'lognorm', 'pearsonV', 'log-logistic', 'weibull').
        - params (tuple): Parameters of the fitted distribution.
        - bins (either a int number or 'auto'): Specification of bins for histogram. Default is 'auto'.
        - smoothing (float, optional): Smoothing factor applied to both observed and expected frequencies. Default is 0.01.

        Returns:
        - chi2_stat (float): The chi-squared test statistic.
        - p_value (float): The p-value of the chi-squared test.
    """
    # Automatically determine the number of bins for observed data
    observed_freq, bin_edges = np.histogram(data, bins=bins)
    num_bins = len(bin_edges) - 1

#   #Generate expected frequencies based on the specified distribution
    if dist == 'gamma':
        expected = np.random.gamma(params[0], scale=params[2], size=len(data))
    elif dist == 'lognorm':
        expected = np.random.lognormal(mean=np.log(params[2]), sigma=params[0], size=len(data))
    elif dist == 'pearsonV':
        expected = params[2] / np.random.gamma(params[0], scale=1/params[0], size=len(data))
    elif dist == 'log-logistic':
        expected = np.random.logistic(params[1], params[2], size=len(data))
    elif dist == 'weibull':
        expected = params[2] * np.random.weibull(params[0], size=len(data))
    else:
        raise ValueError("Invalid distribution specified.")

    # Apply smoothing to both observed and expected frequencies
    observed_freq_smoothed = observed_freq + smoothing
    expected_freq_smoothed, _ = np.histogram(expected, bins=num_bins)
    expected_freq_smoothed = expected_freq_smoothed.astype(float)  # Convert to float
    expected_freq_smoothed += smoothing  # Perform addition operation

    # Compute chi-squared test statistic and p-value
    chi2_stat, p_value = chisquare(observed_freq_smoothed, expected_freq_smoothed)
    return chi2_stat, p_value

def ks_test(data, dist, params):
    """
        Perform Kolmogorov-Smirnov (KS) test to compare empirical data with a theoretical distribution.

        Parameters:
        - data (Pandas series): The empirical data to be tested.
        - dist (str): The name of the distribution ('gamma', 'lognorm', 'weibull', 'log-logistic', 'pearsonV').
        - params (tuple): Parameters of the fitted distribution.

        Returns:
        - ks_stat (float): The test statistic of the KS test.
        - p_value (float): The p-value of the KS test.
    """
    # Perform KS test based on the specified distribution
    if dist == 'gamma':
        return kstest(data, 'gamma', args=params)
    elif dist == 'lognorm':
        return kstest(data, 'lognorm', args=params)
    elif dist == 'weibull':
        return kstest(data, 'weibull_min', args=params)
    elif dist == 'log-logistic':
        return kstest(data, 'fisk', args=params)  # fisk is used for log-logistic
    elif dist == 'pearsonV':
        return kstest(data, 'invgamma', args=params)  # invgamma for Pearson Type V
