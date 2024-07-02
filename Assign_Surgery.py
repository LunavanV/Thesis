import numpy as np
from scipy.stats import fisk

def get_surgery_duration(row):
    """
        Generate surgery duration samples from specified distributions based on parameters in the row.

        Parameters:
        - row (Pandas DataFrame row): Row containing distribution parameters and which distribution needs to be estimated.

        Returns:
        - duration (float): Sampled surgery duration based on the specified distribution.
    """
    #extract distribution type and scale and shape parameters form the row
    dist = row['Distribution_surgery_duration'].iloc[0]
    scale = row['Scale_surgery_duration'].iloc[0]
    shape = row['Shape_surgery_duration'].iloc[0]
    # Depending on the specified distribution type, generate samples and return the estimated surgery duration
    if dist == 'gamma':
        return np.random.gamma(shape, scale=scale, size=None)
    elif dist == 'lognorm':
        return np.random.lognormal(mean=np.log(scale), sigma=shape, size=None)
    elif dist == 'weibull':
        return scale * np.random.weibull(shape, size=None)
    elif dist == 'log-logistic':
        return fisk.rvs(c=shape, scale=scale)
    elif dist == 'pearsonV':
        return scale / np.random.gamma(shape, scale=1/shape, size=None) #the pearsonV distribuiton is the inverse of the gamma

def get_ward_stay(row):
    """
        Generate ward stay length samples from specified distributions based on parameters in the row.

        Parameters:
        - row (Pandas DataFrame row): Row containing distribution parameters and which distribution needs to be estimated.

        Returns:
        - stay_length (float): Sampled ward stay length based on the specified distribution.
    """
    #extract distribution type and scale and shape parameters form the row
    dist = row['Distribution_Length_of_stay'].iloc[0]
    scale = row['Scale_Length_of_stay'].iloc[0]
    shape = row['Shape_Length_of_stay'].iloc[0]
    # Depending on the specified distribution type, generate samples and return the estimated length of stay
    if dist == 'gamma':
        return np.random.gamma(shape, scale=scale, size=None)
    elif dist == 'lognorm':
        return np.random.lognormal(mean=np.log(scale), sigma=shape, size=None)
    elif dist == 'weibull':
        return scale * np.random.weibull(shape, size=None)
    elif dist == 'log-logistic':
        return fisk.rvs(c=shape, scale=scale)
    elif dist == 'pearsonV':
        return scale / np.random.gamma(shape, scale=1/shape, size=None)#the pearsonV distribuiton is the inverse of the gamma

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
                                          Including: The deparment perfomring the surgery, the group within the deparment,
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
    precomputed_values = [(round(row['Duration_Surgery'].iloc[0]), round(row['Wardtime'].iloc[0]),
                           round(increase_surgery_duration*get_surgery_duration(row)), round(increase_ward_stay*get_ward_stay(row))) for row in rows]
    for i, (average_duration_surgery, average_LOS, surgery_duration, total_ward_stay) in enumerate(precomputed_values,start=1):
        if i == 1 and p == 0:
            # the first surgery of the day by the first department
            expected_ending_time = start_time + average_duration_surgery
            arrival_time = start_time - 60 # the patient arrives an hour before surgery
            patient_info_list = [category, schedule[i], surgery_duration, start_time, arrival_time, average_LOS,
                                 average_duration_surgery, total_ward_stay]

        elif i == 1 and p != 0:
            # the first surgery of the day by the second department
            expected_start_time = start_time + 2 * cleaning_time
            arrival_time = expected_start_time - 60 # the patient arrives an hour before surgery
            expected_ending_time = expected_start_time + average_duration_surgery
            patient_info_list = [category, schedule[i], surgery_duration, expected_start_time, arrival_time,
                                 average_LOS, average_duration_surgery, total_ward_stay]

        elif i != 1:
            # the second surgery of a department
            expected_start_time = expected_ending_time + cleaning_time
            arrival_time = expected_start_time - 60 # the patient arrives an hour before surgery
            patient_info_list = [category, schedule[i], surgery_duration, expected_start_time, arrival_time,
                                 average_LOS, average_duration_surgery, total_ward_stay]
            expected_ending_time = expected_start_time + average_duration_surgery
        total_patient_info_list.append(patient_info_list)

    return expected_ending_time, total_patient_info_list
