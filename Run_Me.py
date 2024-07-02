from Model import run_simulation
import pandas as pd
import matplotlib.pyplot as plt
import os
from Visualisation import OT_utilisation, overtime_visualization, Unavailabilty, ward_utilisation
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_dfs(file_pattern, num_files):
    """
    Load multiple dataframes from CSV files based on a file pattern and number of files.

    Parameters:
    - file_pattern (str): The file pattern with a placeholder for the file index.
    - num_files (int): The number of files to load.

    Returns:
    - dfs (list of Pandas DataFrames): List of loaded dataframes.
    """

    # Initialize an empty list to store the dataframes
    dfs = []

    # Iterate over the range of file indices
    for i in range(num_files):
        try:
            # Attempt to read the CSV file using the formatted file pattern
            df = pd.read_csv(file_pattern.format(i))
            # Append the loaded dataframe to the list
            dfs.append(df)
        except pd.errors.EmptyDataError:
            # If the file is empty, skip and continue to the next file
            pass

    # Return the list of loaded dataframes
    return dfs
def run_simulation_wrapper(i, ward_capacity, model_version, schedule_repeats, lower_ward_capacity, increase_surgery_duration, increase_ward_stay, increase_IC_probability):
    """
    Wrapper function to call the run_simulation function with provided parameters.

    Parameters:
    - i (int): Simulation run index.
    - ward_capacity (str): Ward capacity setting.
    - model_version (str): Version of the simulation model.
    - schedule_repeats (int): Number of schedule repeats.
    - lower_ward_capacity (float): Factor to lower the ward capacity.
    - increase_surgery_duration (float): Factor to increase the surgery duration.
    - increase_ward_stay (float): Factor to increase the ward stay duration.
    - increase_IC_probability (float): Factor to increase the intensive care probability.

    Returns:
    - result: Result of the run_simulation function.
    """
    return run_simulation(i, ward_capacity=ward_capacity, model_version=model_version, schedule_repeats=schedule_repeats, lower_ward_capacity=lower_ward_capacity, increase_surgery_duration=increase_surgery_duration, increase_ward_stay=increase_ward_stay, increase_IC_probability=increase_IC_probability)

def running(runs, schedule_repeats, filename, model_version, ward_capacity='low', lower_ward_capacity=1, increase_surgery_duration=1, increase_ward_stay=1, increase_IC_probability=1):
    """
    Run multiple simulation instances and process the results.

    Parameters:
    - runs (int): Number of simulation runs.
    - schedule_repeats (int): Number of schedule repeats.
    - filename (str): Base filename for saving results.
    - model_version (str): Version of the simulation model.
    - ward_capacity (str, optional): Ward capacity setting. Default is 'low'.
    - lower_ward_capacity (float, optional): Factor to lower the ward capacity. Default is 1.
    - increase_surgery_duration (float, optional): Factor to increase the surgery duration. Default is 1.
    - increase_ward_stay (float, optional): Factor to increase the ward stay duration. Default is 1.
    - increase_IC_probability (float, optional): Factor to increase the intensive care probability. Default is 1.

    Returns:
    - None
    """

    # Define the file path and create directories if they do not exist
    filepath = f'results/{filename}'
    excel_dir = os.path.join(filepath, 'excel')
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

    counts_data = []

    # Run simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_simulation_wrapper, i, ward_capacity, model_version, schedule_repeats, lower_ward_capacity, increase_surgery_duration, increase_ward_stay, increase_IC_probability)
            for i in range(runs)
        ]

        # Collect results as they complete
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            df_failed_ward_unavailable_temp, df_failed_OT_unavailable_temp, df_successful_temp, df_resource_use_temp, df_overtime_temp, total_number_of_surgeries, overcapcitation_count = result

            # Save the results to CSV files
            df_failed_ward_unavailable_temp.to_csv(f"{filepath}/excel/failed_ward_unavailable_{i}.csv", index=False)
            df_failed_OT_unavailable_temp.to_csv(f"{filepath}/excel/failed_OT_unavailable_{i}.csv", index=False)
            df_successful_temp.to_csv(f"{filepath}/excel/successful_{i}.csv", index=False)
            df_resource_use_temp.to_csv(f"{filepath}/excel/resource_use_{i}.csv", index=False)
            df_overtime_temp.to_csv(f"{filepath}/excel/overtime_{i}.csv", index=False)

            counts = {
                'Run': i + 1,
                'Successful Surgeries': len(df_successful_temp),
                'Canceled Surgeries (OT Unavailability)': len(df_failed_OT_unavailable_temp),
                'Canceled Surgeries (Ward Unavailability)': len(df_failed_ward_unavailable_temp),
                'Overtime Occurrences': len(df_overtime_temp),
                'Total_number_of_surgeries': total_number_of_surgeries,
                'Times a ward had to overcapacitate': overcapcitation_count
            }
            counts_data.append(counts)

    # Load the results back into lists of DataFrame and analyze them
    df_failed_ward_unavailable_list = load_dfs(f"{filepath}/excel/failed_ward_unavailable_{{}}.csv", runs)
    Unavailabilty(df_failed_ward_unavailable_list, filepath, 'ward')
    plt.close('all')

    df_failed_OT_unavailable_list = load_dfs(f"{filepath}/excel/failed_OT_unavailable_{{}}.csv", runs)
    Unavailabilty(df_failed_OT_unavailable_list, filepath, 'OT')
    plt.close('all')

    df_successful_list = load_dfs(f"{filepath}/excel/successful_{{}}.csv", runs)
    OT_utilisation(df_successful_list, filepath)
    plt.close('all')

    df_overtime_list = load_dfs(f"{filepath}/excel/overtime_{{}}.csv", runs)
    overtime_visualization(df_overtime_list, filepath, schedule_repeats)
    plt.close('all')

    df_resource_list = load_dfs(f"{filepath}/excel/resource_use_{{}}.csv", runs)
    ward_utilisation(df_resource_list, filepath)
    plt.close('all')

    # Save the counts data to an Excel file
    counts_data = pd.DataFrame(counts_data)
    counts_data.to_excel(f"{filepath}/counts.xlsx")

    plt.close()


if __name__ == "__main__":
    # running the mode first determining the number of runs, the number of schedule repeats, file name to same the otuput, which schedule you would want to run
    # other settings are possible to play around with the model.

    running(3, 6 * 13, 'Base_run_A', 'A')
    # running(3, 6 * 13, 'Base_run_B', 'B')
    # running(3, 6 * 13, 'Base_run_C', 'C')
    # running(3, 6 * 13, 'Base_run_D', 'D')
    #
    # running(3, 6 * 13, 'Scenario_1_A', 'A', lower_ward_capacity=0.8, increase_ward_stay=1.1 )
    # running(3, 6 * 13, 'Scenario_1_B', 'B', lower_ward_capacity=0.8, increase_ward_stay=1.1)
    # running(3, 6 * 13, 'Scenario_1_C', 'C', lower_ward_capacity=0.8, increase_ward_stay=1.1)
    # running(3, 6 * 13, 'Scenario_1_D', 'D', lower_ward_capacity=0.8, increase_ward_stay=1.1)
    #
    # running(3, 6 * 13, 'Scenario_2_A', 'A', lower_ward_capacity=0.8, increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_2_B', 'B', lower_ward_capacity=0.8, increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_2_C', 'C', lower_ward_capacity=0.8, increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_2_D', 'D', lower_ward_capacity=0.8, increase_ward_stay=0.9)
    #
    # running(3, 6 * 13, 'Scenario_3_A', 'A', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=1.1)
    # running(3, 6 * 13, 'Scenario_3_B', 'B', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=1.1)
    # running(3, 6 * 13, 'Scenario_3_C', 'C', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=1.1)
    # running(3, 6 * 13, 'Scenario_3_D', 'D', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=1.1)

    # running(3, 6 * 13, 'Scenario_4_A', 'A', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_4_B', 'B', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_4_C', 'C', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=0.9)
    # running(3, 6 * 13, 'Scenario_4_D', 'D', lower_ward_capacity=1.1,ward_capacity = 'high', increase_ward_stay=0.9)

    # running(3, 6 * 13, 'Cross_validation_distributions', '1')
    #
    # running(3, 6 * 13, 'IC_probabliity_0.1', '1', increase_IC_probability = 0.1)
    # running(3, 6 * 13, 'IC_probabliity_0.5', '1', increase_IC_probability = 0.5)
    # running(3, 6 * 13, 'IC_probabliity_2', '1', increase_IC_probability = 2)
    # running(3, 6 * 13, 'IC_probabliity_10', '1', increase_IC_probability = 10)

    # running(3, 6 * 13, 'Surgery_duration_0.1', '1',increase_surgery_duration = 0.1)
    # running(3, 6 * 13, 'Surgery_duration_0.5', '1',increase_surgery_duration = 0.5)
    # running(3, 6 * 13, 'Surgery_duration_2', '1',increase_surgery_duration = 2)
    # running(3, 6 * 13, 'Surgery_duration_10', '1', increase_surgery_duration = 10)
    #
    # running(3, 6 * 13, 'Ward_capacity_0.1', '1', lower_ward_capacity=0.1)
    # running(3, 6 * 13, 'Ward_capacity_0.5', '1', lower_ward_capacity=0.5)
    # running(3, 6 * 13, 'Ward_capacity_2', '1', lower_ward_capacity=2)
    # running(3, 6 * 13, 'Ward_capacity_10', '1', lower_ward_capacity=10)
    #
    # running(3, 6 * 13, 'Ward_stay_0.1', '1', increase_ward_stay=0.1)
    # running(3, 6 * 13, 'Ward_stay_0.5', '1', increase_ward_stay=0.5)
    # running(3, 6 * 13, 'Ward_stay_2', '1', increase_ward_stay=2)
    # running(3, 6 * 13, 'Ward_stay_10', '1', increase_ward_stay=10)











