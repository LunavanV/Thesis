import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

def ward_utilisation(df_resource_use, filepath):
    """
   Calculate and visualize ward utilisation from a list of resource use dataframes and save the plots to files.

   Parameters:
   - df_resource_use (list of Pandas DataFrames): List of dataframes containing resource use data.
   - filepath (str): Path to save the resulting plots.

   Returns/output:
    - A plot with average utilisation for all the wards in the hospital across the 28-day schedule cycle
    - A plot with average utilisation per ward in the hospital
    - A plot per ward with the utilisation across 28-day schedule cycle
   """

    # Assign a run number to each dataframe
    for i, df in enumerate(df_resource_use):
        df['run'] = i + 1

    # Concatenate all dataframes into a single dataframe
    df = pd.concat(df_resource_use)
    # Group by simulation_day and run, summing capacity and claimed quantities
    grouped_df = df.groupby(['simulation_day', 'run']).sum().reset_index()

    # Calculating the total utilisation per day for each ward and for the hospital
    utilisation_data = []
    total_utilisation_data = []
    wards = [col.split('_capacity')[0] for col in df.columns if 'capacity' in col]

    for index, row in grouped_df.iterrows():
        simulation_day = row['simulation_day']
        run = row['run']
        schedule_day = int((simulation_day % 28) + 1)

        # Calculate total capacity and total claimed quantities
        total_capacity = sum(row[f"{ward}_capacity"] for ward in wards)
        total_claimed = sum(row[f"{ward}_claimed_quantity"] for ward in wards)

        # Calculate total utilisation
        total_utilisation = 1 if total_capacity == 0 and total_claimed > 0 else 0 if total_claimed == 0 else total_claimed / total_capacity
        total_utilisation_data.append([simulation_day, run, schedule_day, total_utilisation])

        # Calculate utilisation for each ward
        for ward in wards:
            capacity_col = f"{ward}_capacity"
            claimed_col = f"{ward}_claimed_quantity"
            capacity = row[capacity_col]
            claimed_quantity = row[claimed_col]
            utilisation = 1 if capacity == 0 and claimed_quantity > 0 else 0 if claimed_quantity == 0 else claimed_quantity / capacity
            utilisation_data.append([simulation_day, run, schedule_day, ward, utilisation])

    # Create dataframes for utilisation data
    utilisation_df = pd.DataFrame(utilisation_data, columns=['simulation_day', 'run', 'schedule_day', 'ward', 'utilisation'])
    total_utilisation_df = pd.DataFrame(total_utilisation_data, columns=['simulation_day', 'run', 'schedule_day', 'utilisation'])

    # Ensure we handle NaN or Inf values
    utilisation_df = utilisation_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['utilisation'])
    total_utilisation_df = total_utilisation_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['utilisation'])

    # Find the maximum utilisation value for setting plot limits
    max_value = utilisation_df['utilisation'].max()
    if np.isnan(max_value) or np.isinf(max_value):
        max_value = 1  # Default to 1 if max_value is not valid
    # Create boxplots for each ward
    fig, axs = plt.subplots(2, 4, figsize=(36, 18))

    for i, ward in enumerate(wards):
        ax = axs[i // 4, i % 4]
        sns.boxplot(data=utilisation_df[utilisation_df['ward'] == ward], x='schedule_day',
                    y='utilisation', ax=ax)
        ax.set_title(f'{ward} utilisation per Schedule Day')
        ax.set_xlabel('Schedule Day')
        ax.set_ylabel('utilisation')
        ax.set_ylim(0, min(2, 0.9*max_value))  # Set y-axis limit from 0 to max value

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    # Save the plot for each ward utilisation
    plt.savefig(f'{filepath}/ward_utilisation_per_ward_per_scheduleday.png')  # Save the plot

    # Create a new figure for the total hospital utilisation plot
    plt.figure(figsize=(8, 6))
    # Create a boxplot for total hospital utilisation per schedule day
    sns.boxplot(data=total_utilisation_df, x='schedule_day', y='utilisation')
    plt.title('Total Hospital utilisation per Schedule Day')
    plt.xlabel('Schedule Day')
    plt.ylabel('utilisation')
    plt.ylim(0, max(2,total_utilisation_df['utilisation'].max()))  # Set y-axis limit from 0 to max value

    # Adjust layout for better spacing
    plt.tight_layout()
    # Save the plot for total hospital utilisation
    plt.savefig(f'{filepath}/ward_utilisation_total.png')

    # Create a new figure for the total hospital utilisation plot
    plt.figure(figsize=(8, 6))
    # Create a boxplot for total hospital utilisation per schedule day
    sns.boxplot(data=utilisation_df, x='ward', y='utilisation')
    plt.title('Total Hospital utilisation per ward')
    plt.xlabel('Ward name')
    plt.ylabel('utilisation')
    plt.ylim(0, max(2, utilisation_df['utilisation'].max()))  # Set y-axis limit from 0 to max value

    # Adjust layout for better spacing
    plt.tight_layout()
    # Save the plot for total hospital utilisation
    plt.savefig(f'{filepath}/ward_utilisation_per_ward.png')

    avg_utilisation_per_run = total_utilisation_df.groupby('run')['utilisation'].mean().reset_index(name='average_utilisation')
    return avg_utilisation_per_run

def OT_utilisation(df_surgeries_list, filepath):
    """
    Calculate and visualize OT utilisation from a list of surgery dataframes and save the plots to files.

    Parameters:
    - df_surgeries_list (list of Pandas DataFrames): List of dataframes containing surgery data.
    - filepath (str): Path to save the resulting plots.

    Returns:
    - A plot with average utilisation for all the OTs in the hospital across the 28-day schedule cycle
    - A plot with average utilisation per OT in the hospital
    - A plot per OT with teh utilisation across 28-day schedule cycle
    """
    # Assign a run number to each dataframe
    for i, df in enumerate(df_surgeries_list):
        df['run'] = i + 1

    # Concatenate all dataframes into a single dataframe
    df = pd.concat(df_surgeries_list)

    # Group by OT, simulation_day, and run, summing surgery durations
    df_sum = df.groupby(['OT', 'simulation_day', 'run'])['surgery_duration'].sum().reset_index(name='total_surgery_duration')

    # Calculate the utilisation
    df_sum['utilisation'] = df_sum['total_surgery_duration'] / 450
    df_sum['schedule_day'] = (df_sum['simulation_day'] % 28) + 1

    OTs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'MRI']

    # Find the maximum utilisation value for setting plot limits
    max_value = max(df_sum['utilisation'])
    if np.isnan(max_value) or np.isinf(max_value):
        max_value = 1  # Default to 1 if max_value is not valid

    # Create boxplots for each OT
    fig, axs = plt.subplots(3, 4, figsize=(36, 18))
    df_sum['OT'] = df_sum['OT'].astype(str)
    for i, ot in enumerate(OTs):
        ax = axs[i // 4, i % 4]
        # Create a boxplot for the utilisation of each OT per schedule day
        sns.boxplot(data=df_sum[df_sum['OT'] == ot], x='schedule_day', y='utilisation', ax=ax)
        ax.set_title(f'OT {ot} utilisation per Schedule Day')
        ax.set_xlabel('Schedule Day')
        ax.set_ylabel('utilisation')
        ax.set_ylim(0, min(2, 0.9*max_value)) # Set y-axis limit
        ax.set_xlim(0,28) # Set x-axis limit to the number of days in the cycle

    # Remove empty subplots
    for j in range(len(OTs), 12):
        fig.delaxes(axs.flatten()[j])

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    # Save the plot for each OT utilisation
    plt.savefig(f'{filepath}/OT_utilisation_per_OT_per_scheduleday')  # Save the plot

    # Create a new figure for the total hospital utilisation plot
    plt.figure(figsize=(8, 6))
    # Create a boxplot for total hospital utilisation per schedule day
    sns.boxplot(data=df_sum, x='schedule_day', y='utilisation')
    plt.title('Total OT utilisation per Schedule Day')
    plt.xlabel('Schedule Day')
    plt.ylabel('utilisation')
    plt.ylim(0, max(2, min(5,df_sum['utilisation'].max())))  # Set y-axis limit from 0 to 2

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot for total hospital utilisation
    plt.savefig(f'{filepath}/OT_utilisation_total.png')  # Save the plot

    # Create a new figure for the total hospital utilisation plot
    plt.figure(figsize=(8, 6))
    # Create a boxplot for total hospital utilisation per schedule day
    sns.boxplot(data=df_sum, x='OT', y='utilisation')
    plt.title('Total OT utilisation per OT')
    plt.xlabel('OT name')
    plt.ylabel('utilisation')
    plt.ylim(0, max(2, min(5,df_sum['utilisation'].max())))  # Set y-axis limit from 0 to 2

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot for total hospital utilisation
    plt.savefig(f'{filepath}/OT_utilisation_per_OT.png')  # Save the plot

    avg_utilisation_per_run = df_sum.groupby('run')['utilisation'].mean().reset_index(name='average_utilisation')
    return avg_utilisation_per_run

def overtime_visualization(df_overtime_list, filepath, schedule_repeats):
    """
    Visualize overtime frequency and length from a list of dataframes and save the plots to a file.

    Parameters:
    - df_overtime_list (list of Pandas DataFrames): List of dataframes containing overtime data.
    - filepath (str): Path to save the resulting plots.
    - schedule_repeats (int): Number of schedule repeats to calculate total surgery days.

    Returns:
    - A plot per OT that includes the frequency and length of the overtime and a histogram with number of overtime
    occurrances and frequency of length occurances.
    - A plot for the total hospital that includes the occurances of overtime per OT and the occurences of overtime per
    duration of overtime
    """
    # Initialize lists to store frequency, average length, and raw length data
    frequency_data = []
    total_frequency_data = []
    length_data = []
    raw_length_data = []
    num_runs = len(df_overtime_list)

    # Iterate over each run's dataframe
    for df in df_overtime_list:
        # Convert 'OT' column to string
        df['OT'] = df['OT'].astype(str)
        # Add one to every element of schedule day to adjust the day numbering
        df['schedule_day'] += 1

        # Group by OT and schedule_day and count the occurrences of overtime
        grouped = df.groupby(['OT', 'schedule_day']).size().reset_index(name='frequency')
        frequency_data.append(grouped)

        # Group by OT and schedule_day and count the occurrences of overtime
        grouped = df.groupby(['OT']).size().reset_index(name='frequency')
        total_frequency_data.append(grouped)

        # Group by OT and schedule_day and calculate the mean of overtime length
        length_grouped = df.groupby(['OT', 'schedule_day'])['overtime'].mean().reset_index(name='avg_length')
        length_data.append(length_grouped)

        # Collect raw overtime data for histogram
        raw_length_data.append(df[['OT', 'schedule_day', 'overtime']])

    # Combine frequency and length data from all runs
    combined_frequency_data = pd.concat(frequency_data)
    combined_total_frequency_data = pd.concat(total_frequency_data)
    combined_length_data = pd.concat(length_data)
    combined_raw_length_data = pd.concat(raw_length_data)

    # Get unique OT values
    unique_ot_values = sorted(combined_raw_length_data['OT'].unique())
    # Calculate the total number of surgery days
    total_surgery_days = 20 * schedule_repeats

    # Determine axis limits for plotting
    max_frequency = combined_frequency_data['frequency'].max()
    max_avg_length = combined_length_data['avg_length'].max()
    max_overtime_length = combined_raw_length_data['overtime'].max()

    # Create bin edges for histogram with a size of 15 minutes
    bin_edges = np.arange(0, max_overtime_length + 15, 15)

    # Create subplots for each OT
    fig, axes = plt.subplots(len(unique_ot_values), 3, figsize=(30, 5 * len(unique_ot_values)))

    for i, ot in enumerate(unique_ot_values):
        # Filter data for the current OT
        ot_freq_data = combined_frequency_data[combined_frequency_data['OT'] == ot]
        ot_length_data = combined_length_data[combined_length_data['OT'] == ot]
        ot_raw_length_data = combined_raw_length_data[combined_raw_length_data['OT'] == ot]
        days = sorted(combined_raw_length_data['schedule_day'].unique())

        # Ensure data for all days is included, with empty arrays for missing days
        frequency_boxplot_data = [ot_freq_data[ot_freq_data['schedule_day'] == day]['frequency'].values
                                  if day in ot_freq_data['schedule_day'].values else [] for day in days]
        avg_length_boxplot_data = [ot_length_data[ot_length_data['schedule_day'] == day]['avg_length'].values
                                   if day in ot_length_data['schedule_day'].values else [] for day in days]

        # Plot frequency boxplots
        axes[i, 0].boxplot(frequency_boxplot_data, positions=days)
        axes[i, 0].set_ylim(0, max_frequency)
        axes[i, 0].set_title(f'Frequency Boxplot for OT {ot}')
        axes[i, 0].set_xlabel('Schedule Day')
        axes[i, 0].set_ylabel('Frequency')

        # Plot average length boxplots
        axes[i, 1].boxplot(avg_length_boxplot_data, positions=days)
        axes[i, 1].set_ylim(0, max_avg_length)
        axes[i, 1].set_title(f'Average Length Boxplot for OT {ot}')
        axes[i, 1].set_xlabel('Schedule Day')
        axes[i, 1].set_ylabel('Average Overtime Length')

        # Plot histogram of overtime lengths for each OT
        ot_length_data = ot_raw_length_data['overtime']
        counts, bins, patches = axes[i, 2].hist(ot_length_data, bins=bin_edges, edgecolor='black')
        percentages = (counts / (total_surgery_days*num_runs)) * 100
        for count, patch in zip(percentages, patches):
            patch.set_height(count)
        axes[i, 2].axvline(x=45, color='red', linestyle='--')
        axes[i, 2].set_title(f'Histogram of Overtime Length for OT {ot}')
        axes[i, 2].set_xlabel('Length of Overtime (min)')
        axes[i, 2].set_ylabel('Percentage of Surgery Days')
        axes[i, 2].set_ylim(0, max(max(percentages) * 1.1, 5))  # Add some space for better visualization

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    # Save the plots to a file
    plt.savefig(f'{filepath}/Overtime')

    # Set up the figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # First subplot (left): Frequency Boxplot for All OTs
    combined_total_frequency_data['percentages'] = (combined_total_frequency_data['frequency'] / (
        total_surgery_days)) * 100
    frequency_boxplot_data = [
        combined_total_frequency_data[combined_total_frequency_data['OT'] == ot]['percentages'].values for ot in
        unique_ot_values]
    axs[0].boxplot(frequency_boxplot_data)
    axs[0].set_xticks(range(1, len(unique_ot_values) + 1))
    axs[0].set_xticklabels(unique_ot_values)
    axs[0].set_ylim(0, max(combined_total_frequency_data['percentages'].max(),
                           25))  # Set the y-axis limit to the highest value or 25
    axs[0].set_title('Frequency Boxplot for All OTs')
    axs[0].set_xlabel('OT')
    axs[0].set_ylabel('Frequency')

    # Second subplot (right): Histogram of Overtime Length for OTs
    or_length_data = combined_raw_length_data['overtime']
    counts, bins, patches = axs[1].hist(or_length_data, bins=bin_edges, edgecolor='black')
    percentages = (counts / (total_surgery_days *num_runs* len(unique_ot_values))) * 100  # Not only dividing by the total number of days in the schedule but also the number of OTs
    for count, patch in zip(percentages, patches):
        patch.set_height(count)
    axs[1].axvline(x=45, color='red', linestyle='--')
    axs[1].set_title('Histogram of Overtime Length for OTs')
    axs[1].set_xlabel('Length of Overtime (min)')
    axs[1].set_ylabel('Percentage of Surgery Days')
    axs[1].set_ylim(0, max(5, max(percentages) * 1.1))  # Set limit to the higher value or 25


    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{filepath}/Total_overtime.png')  # Save the plot

def Unavailabilty(df_failed_unavailable_list, filepath, sort):
    """
    Calculate and analyze the unavailability counts from a list of dataframes that contains the occurances of a surgery
    being cancelled because a ward or OT is not available and save the results to an Excel file.

    Parameters:
    - df_failed_unavailable_list (list of Pandas DataFrames): List of dataframes containing unavailability data.
    - filepath (str): Path to save the resulting Excel file.
    - sort (str): Column name to be used for sorting and calculating counts.

    Returns/output:
    - Excel file of the counted dataframe, for either sort.
    """
    # Initialize an empty list to store the counts for each dataframe
    counts_list = []
    # Loop through each dataframe and calculate the counts
    for i, df in enumerate(df_failed_unavailable_list):
        # Ensure the 'sort' column contains individual elements instead of lists
        df[sort] = df[sort].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Calculate the value counts for the either OT or ward unavailability occurrances and reset the index
        df_counts = df[sort].value_counts().reset_index()
        # Rename the columns for clarity
        df_counts.columns = [sort, f'count_{i}']
        #convert names to strings
        df_counts[sort]=df_counts[sort].astype(str)
        # Append the counts dataframe to the counts_list
        counts_list.append(df_counts)

    # Check if there are any dataframes to process
    if len(counts_list) <1:
        merged_counts = pd.DataFrame()
    else:
        # Initialize merged_counts with the first counts dataframe
        merged_counts = counts_list[0]

        # Merge all the counts dataframes
        for df_counts in counts_list[1:]:
            merged_counts = pd.merge(merged_counts, df_counts, on=sort, how='outer').fillna(0)

        # Calculate the average, standard deviation, and variance of the counts
        count_columns = [col for col in merged_counts.columns if col.startswith('count_')]
        merged_counts['average'] = merged_counts[count_columns].mean(axis=1)
        merged_counts['std_dev'] = merged_counts[count_columns].apply(np.std, axis=1)
        merged_counts['variance'] = merged_counts[count_columns].apply(np.var, axis=1)

        # Drop the individual count columns as they are no longer needed
        merged_counts = merged_counts.drop(columns=count_columns)

        # Calculate the percentage of the total for the average
        total_average = merged_counts['average'].sum()
        merged_counts['percentage'] = (merged_counts['average'] / total_average) * 100

    # Save the final dataframe to an Excel file
    merged_counts.to_excel(f'{filepath}/unavailability_{sort}.xlsx')