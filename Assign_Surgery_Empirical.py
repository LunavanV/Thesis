import pandas as pd
import random

df = pd.read_excel('Input_Data/data_with_group_number.xlsx')
df_test = df[df['train-test'] == 'test'].copy()
# Convert 'Group_number' column to string type
df_test['Group_number'] = df_test['Group_number'].astype(str)

def assign_surgery(schedule, df, OT, start_time, p=0, increase_surgery_duration = 1, increase_ward_stay =1):
    """
        Taking the timeslots for that day and assigning the inforamtion about the patient including the duration of the
        surgery and the length of stay.

        Parameters:
        - schedule (list): List containing the surgical schedule, the first element is the category that needs to be
                           assigned, and the following are the group numbers within that category.
        - df (Pandas DataFrame): DataFrame containing parameters and averages about surgery and ward stay.
        - OT (int): Operating theater identifier or number.
        - start_time (int): The assigned starting time, either opening of the OT or determined by previous surgeries
        - p (int, optional): Parameter indicating if there was a different department that has used the OT before on the
                             same day, indicating the need for extra cleaning time.
        - increase_surgery_duration (float, optional): Factor to increase surgery duration, used for sensitivity analysis.
        - increase_ward_stay (float, optional): Factor to increase ward stay duration, used for sensitivity analysis.

        Returns:
        - expected_ending_time (int): Expected time when the last surgery ends.
        - total_patient_info_list (list): List of lists containing detailed patient information for each scheduled surgery.
                                          Including: The department performing the surgery, the group within the department,
                                          the duration of the surgery (in minutes), the starting time of the surgery (in minutes of the day),
                                          the arrival time of the patient (in minutes of the day), the average length of
                                          stay of a patient of that category and group, the average surgery duration of
                                          stay of a patient of that category and group and the assigned ward stay for the patient.
    """
    cleaning_time = 15  # Regular cleaning time is set to 15 minutes
    category = schedule[0]
    total_patient_info_list = []
    # Precompute values for each surgery in the schedule
    rows = [df[(df['Category'] == category) & (df['Group_number'] == group)] for group in schedule[1:]]
    # precomputed_values = [(round(row['Duration_Surgery'].iloc[0]), round(row['Wardtime'].iloc[0]),
    #                        round(increase_surgery_duration*get_surgery_duration(row)), round(increase_ward_stay*get_ward_stay(row))) for row in rows]
    precomputed_values = [(round(row['Duration_Surgery'].iloc[0]), round(row['Wardtime'].iloc[0])) for row in rows]
    # for i, (average_duration_surgery, average_LOS, surgery_duration, total_ward_stay) in enumerate(precomputed_values,start=1):
    for i, (average_duration_surgery, average_LOS) in enumerate(precomputed_values,start=1):
        if i == 1 and p == 0:
            surgery_duration, total_ward_stay = new_get_surgery_duration(category, schedule[i])


            # the first surgery of the day by the first department
            expected_ending_time = start_time + average_duration_surgery
            arrival_time = start_time - 60 # the patient arrives an hour before surgery
            patient_info_list = [category, schedule[i], surgery_duration, start_time, arrival_time, average_LOS,
                                 average_duration_surgery, total_ward_stay]

        elif i == 1 and p != 0:
            surgery_duration, total_ward_stay = new_get_surgery_duration(category, schedule[i])

            # the first surgery of the day by the second department
            expected_start_time = start_time + 2 * cleaning_time
            arrival_time = expected_start_time - 60 # the patient arrives an hour before surgery
            expected_ending_time = expected_start_time + average_duration_surgery
            patient_info_list = [category, schedule[i], surgery_duration, expected_start_time, arrival_time,
                                 average_LOS, average_duration_surgery, total_ward_stay]

        elif i != 1:
            surgery_duration, total_ward_stay = new_get_surgery_duration(category, schedule[i])

            # the second surgery of a department
            expected_start_time = expected_ending_time + cleaning_time
            arrival_time = expected_start_time - 60 # the patient arrives an hour before surgery
            patient_info_list = [category, schedule[i], surgery_duration, expected_start_time, arrival_time,
                                 average_LOS, average_duration_surgery, total_ward_stay]
            expected_ending_time = expected_start_time + average_duration_surgery
        total_patient_info_list.append(patient_info_list)

    return expected_ending_time, total_patient_info_list

def new_get_surgery_duration(category, group_number):
    """
    Get the surgery duration and length of stay for a given category and group number.

    Parameters:
    - category (str): The category of the surgery.
    - group_number (int): The group number of the surgery.

    Returns:
    - Surgery_duration (float): Duration of the surgery.
    - length_of_stay (float): Length of stay in the ward.
    """

    group_number = str(group_number)
    global df_test

    # Apply strip operation directly to ensure no leading/trailing spaces in 'Group_number'
    df_test['Group_number'] = df_test['Group_number'].apply(lambda x: x.strip())

    # Filter the dataframe for the specified group number and category
    df_group = df_test[(df_test['Group_number'] == group_number) & (df_test['Category'] == category)].copy()

    # Randomly choose an entry from the filtered dataframe
    chosen = random.randint(0, len(df_group) - 1)

    # Get the surgery duration and length of stay from the chosen entry
    Surgery_duration = df_group.iloc[chosen]['Duration_Surgery']
    length_of_stay = df_group.iloc[chosen]['Wardtime']

    return Surgery_duration, length_of_stay

