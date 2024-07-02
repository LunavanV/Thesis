import pandas as pd
import numpy as np
import json

# Create a DataFrame with columns for day and minute
days = []
minutes = []
# Loop through 7 days (0 to 6)
for day in range(7):
    # Loop through 1440 minutes in a day (0 to 1439)
    for minute in range(1440):
        # Append current day and minute to respective lists
        days.append(day)
        minutes.append(minute)

# Create a dictionary with 'Day' and 'Minute' as keys and days and minutes lists as values
data = {'Day': days, 'Minute': minutes}

# Create a DataFrame 'df' using the dictionary 'data' with the day and minutes
df = pd.DataFrame(data)

# Define the ick_value function to handle different conditions based on both day and number
def ick_value(day, number):
    """
    Determine the ICK value based on the day and the type of ICK.

    Parameters:
    - day (int): Day of the week (0 to 6, where 0 is Monday).
    - number (int): Number indicating the ICK category (1 or 2) where 1 is the vented ICK and 2 is the ICK without venting

    Returns:
    - float: ICK value corresponding to the given day and number.
    """
    if day in [0, 1, 2, 3, 4]: #monday until friday
        # ICK 2 and 3 with venting
        if number == 1:
            return 4
        else:
            # ICK 1 and 4 without venting
            return 4.5
    else:
        #for saturday until sunday
        if number == 1:
            #Value for ICK 2/3
            return 4
        else:
            #Value for ICK 1/4
            return 3.5
# Apply the function with different parameters to create multiple "ICK" columns
df['ICK2_3'] = df.apply(lambda row: ick_value(row['Day'], 1), axis=1)
df['ICK1_4'] = df.apply(lambda row: ick_value(row['Day'], 2), axis=1)

def MCU_value(day, minute, department):
    """
    Calculate the capacity of the different MCUs based on the day and time. Each MCU has its own capacity (one MCU is excluded)

    Parameters:
    - day (int): Day of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday).
    - minute (int): Minute of the day.
    - department (str): Department identifier ('KZC', 'KTC', 'MCKG', 'SK4SP4').

    Returns:
    - float: Capacity value corresponding to the given parameters.
    """
    if day in [0, 1, 2, 3, 4]:# Monday to Friday
        if (minute >= 0 and minute < 419) or (minute >= 1379 and minute < 1440):
            # Early morning and late night
            if department == 'KZC':
                return 9
            if department == 'KTC':
                return 4
            if department == 'MCKG':
                return 3
            if department == 'SK4SP4':
                return 3.4
        elif minute >= 420 and minute <= 959:
            # Morning
            if department == 'KZC':
                return 10
            if department == 'KTC':
                return 4.5
            if department == 'MCKG':
                return 3.5
            if department == 'SK4SP4':
                return 4
        else:
            # Afternoon and evening
            if department == 'KZC':
                return 9.5
            if department == 'KTC':
                return 4.5
            if department == 'MCKG':
                return 3.5
            if department == 'SK4SP4':
                return 3.4
    else: # Saturday and Sunday
        if (minute >= 0 and minute < 419) or (minute >= 1380 and minute <= 1440):
            # Early morning and late night
            if department == 'KZC':
                return 8
            if department == 'KTC':
                return 3
            if department == 'MCKG':
                return 2.5
            if department == 'SK4SP4':
                return 2.2
        elif minute >= 419 and minute < 959:
            # Morning
            if department == 'KZC':
                return 8.5
            if department == 'KTC':
                return 3
            if department == 'MCKG':
                return 2.5
            if department == 'SK4SP4':
                return 3.3
        else:
            # Afternoon and evening
            if department == 'KZC':
                return 8
            if department == 'KTC':
                return 3
            if department == 'MCKG':
                return 2.5
            if department == 'SK4SP4':
                return 2.2

# Apply the function to set the capacity for each ward
df['KCZ'] = df.apply(lambda row: MCU_value(row['Day'], row['Minute'], 'KZC'), axis=1)
df['KTC'] = df.apply(lambda row: MCU_value(row['Day'], row['Minute'], 'KTC'), axis=1)
df['MCKG'] = df.apply(lambda row: MCU_value(row['Day'], row['Minute'], 'MCKG'), axis=1)
df['SK4SP4'] = df.apply(lambda row: MCU_value(row['Day'], row['Minute'], 'SK4SP4'), axis=1)
def KCN_value(day, minute):
    """
    Calculate the capacity of the KCN MCU based on the day and time, split up from the other because there is more
    variation. Each MCU has its own capacity (one MCU is excluded)

    Parameters:
    - day (int): Day of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday).
    - minute (int): Minute of the day.

    Returns:
    - float: Capacity value corresponding to the given parameters.
    """
    if day in [0, 3, 4]:
        if (minute >= 0 and minute <419) or (minute >= 1380 and minute <= 1440):
            return 7.5
        elif minute >= 419 and minute < 959:
            return 8.5
        else:
            return 8
    elif day in [2,5]:
        if (minute >= 0 and minute < 419) or (minute >= 1380 and minute <= 1440):
            return 7.5
        elif minute >= 419 and minute < 959:
            return 10.5
        else:
            return 8
    else:
        if (minute >= 0 and minute <= 419) or (minute >= 1380 and minute <= 1440):
            return 4.5
        elif minute >= 419 and minute < 959:
            return 5
        else:
            return 4.5

# Apply the function to create the "KCZ" capacity
df['KCN'] = df.apply(lambda row: KCN_value(row['Day'], row['Minute']), axis=1)

def daycare_value(day, minute):
    """
   Calculate the daycare capacity based on the given day and minute.

   Parameters:
   - day (int): Day of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday).
   - minute (int): Minute of the day (0 to 1439).

   Returns:
   - int: Daycare value corresponding to the given day and minute.
    """
    if day in [0, 1, 2, 3, 4]:# Monday to Friday
        if minute >= 419 and minute < 599:
            return 8
        elif minute >= 599 and minute < 839:
            return 9
        elif minute >= 839 and minute < 1079:
            return 5
        else:
            return 0
    else: #closed on weekends
        return 0

# Apply the function to create the "KCZ" column
df['Daycare'] = df.apply(lambda row: daycare_value(row['Day'], row['Minute']), axis=1)
df_ward = df

# Create a dictionary with 'Day' and 'Minute' as keys and days and minutes lists as values
data = {'Day': days, 'Minute': minutes}
# Create a DataFrame 'df' using the dictionary 'data' with the day and minutes
df_OT = pd.DataFrame(data)
def OT_availability(day, minute):
    """
    Determine the availability of OT (Operating Theater) based on the given day and minute. Operating theaters open at 8
    in the morning and close at 16:30. They are only opend on weekdays

    Parameters:
    - day (int): Day of the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday).
    - minute (int): Minute of the day (0 to 1439).

    Returns:
    - int: Availability status of OT (0 for unavailable, 1 for available).
    """
    if day in [0, 1, 2, 3, 4] and minute >= 479 and minute <= 929: #from monday until friday opened from 8 till 16:30
        return 1
    else:
        return 0

# Apply the function to get OT availability for the numbered OTs
for i in range(1, 11):
    df_OT[str(i)] = df_OT.apply(lambda row: OT_availability(row['Day'], row['Minute']), axis=1)

# Apply the function to get OT availability for the MRI
df_OT['MRI'] = df_OT.apply(lambda row: OT_availability(row['Day'], row['Minute']), axis=1)

def setup_schedule_repeats(df, repeats):
    """
    Repeat the rows of a DataFrame to set up availability schedule for the total duration of the simulation

    Parameters:
    - df (pandas DataFrame): The DataFrame containing the schedule to repeat.
    - repeats (int): Number of times to repeat the DataFrame.

    Returns:
    - pandas DataFrame: DataFrame with rows repeated according to the specified number of repeats.
    """

    df_copy = df.copy()# Make a copy of the original DataFrame
    for i in range(repeats-1):
        # Concatenate the copy to the original DataFrame for the prespecified number of times
        df = pd.concat([df, df_copy], ignore_index=True)
    return df

def load_json(cell_value): #to turn all the values into lists
    """
   Convert a JSON string into a Python list or handle NaN values so the schedule that is imported is in the right format

   Parameters:
   - cell_value (str or any): Value to convert. If it's a JSON string, it will be converted to a Python list.
                              If it's NaN, it will be returned as None.
                              Otherwise, it will be returned unchanged.

   Returns:
   - list or None or any: Converted Python list if input was a JSON string, None if input was NaN,
                         or the input value unchanged if it was neither a JSON string nor NaN.
   """
    if isinstance(cell_value, str):  # Check if the cell_value is a string
        cell_value = json.loads(cell_value) # Convert JSON string to Python list
        return cell_value
    elif pd.isna(cell_value):  # Check if the value is NaN
        return None # Return None if cell_value is NaN
    else:
        return cell_value # Return cell_value unchanged if it's neither a string nor NaN
