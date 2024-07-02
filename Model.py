import salabim as sim
import pandas as pd
import numpy as np
import random
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

#importing functions that have been created in other python files
from Ward_OT_Scheduling import df_ward, df_OT, load_json, setup_schedule_repeats
from set_resource_capacity import set_resource_capacity
from Assigning_wards import assigning_ward
from ICK_Assignment import Assining_ICK
from Assign_Surgery import assign_surgery
class TimestepAction(sim.Component):
    """
       A class to represent a time step action in the simulation environment. This class is responsible for setting the
       resource capacity at each time step and collecting resource usage data.

       Methods:
       - process(): Main process that the component will execute.
       """

    def process(self):
        """
        Main process to set resource capacity and collect resource usage data at each time step.

        This process initializes resource data collection, sets the resource capacity,
        and periodically updates and collects resource usage data.

        """

        # Initialize an empty list to store resource usage data
        self.env.resource_data =[]

        # Initial hold to synchronize the start time, make sure that it is updated on the hour
        self.hold(59)

        # Set the initial resource capacity
        set_resource_capacity(self.env)

        # Main loop to periodically update and collect resource usage data
        while True:

            # Update the resource capacity at the current time step
            set_resource_capacity(self.env)

            # Create a row to store the resource data for the current day
            row = {'schedule_day': self.env.days%28, 'simulation_day': self.env.days}

            # Iterate over each resource and ward to collect capacity and claimed quantity
            for resource, ward in self.env.resource_wards.items():
                row[f'{resource}_capacity'] = ward.capacity()
                row[f'{resource}_claimed_quantity'] = ward.claimed_quantity()

            # Append the collected data to the resource data list
            self.env.resource_data.append(row)

            # Hold for 60 time units (minutes) before the next update
            self.hold(60) #can be adjusted to get more detailed information and updating, make the model signficantly slower
class day_step(sim.Component):
    """
    A class to set up the daily schedules in the simulation enviroment and update the day count. Every day it takes the
    current day of the simulation in the schedule and for each OT determines which surgeries need to be performed. Then
    it uses a separate function to set up all the information on the surgeries and initiates the entity of patient in the
    enviorment.

    Methods:
    - setup(df_schedule_surgery): Sets up the surgery schedule DataFrame.
    - process(): Main process that the component will execute to handle daily schedules.
    """

    def setup(self, df_schedule_surgery):
        """
        collect the surgery schedule setup by the enviroment so it can be accessed by the class

        Parameters:
        - df_schedule_surgery (DataFrame): DataFrame containing the surgery schedules.

        """
        self.df_schedule_surgery = df_schedule_surgery

    def process(self):
        """
        Main process to handle the daily surgery schedules.This process checks the current day, retrieves the
        corresponding surgery schedule, and creates patients based on the schedule.

        """
        while True:
            # Determine the current day in the 28-day cycle
            current_day = self.env.days % 28

            # Check if the current day is a day when the OT is open
            if current_day in self.df_schedule_surgery.iloc[:, 0].values:
                # Get the row corresponding to the current day
                current_row = self.df_schedule_surgery[self.df_schedule_surgery.iloc[:, 0] == current_day].iloc[0]

                # Precompute patient creation function to avoid repetitive function calls
                create_patient = lambda info, OT, second_category=False: patients(
                    category=info[0], group=info[1], OT=OT, surgery_duration=info[2],
                    start_time=info[3], arrival_time=info[4], average_LOS=info[5],
                    average_duration_surgery=info[6], total_ward_stay=info[7], second_category=second_category
                )

                # Loop through all the OTs (Operating Theatres)
                for i in range(1, len(current_row)):
                    current_cell = current_row.iloc[i]
                    OT_column = self.df_schedule_surgery.columns[i]

                    # Check if the OT has surgeries assigned
                    if current_cell is None:
                        continue

                    # Check if it is only one specialty that takes place in that OT
                    if not isinstance(current_cell[0], list):
                        ending_time, patient_info = assign_surgery(current_cell, self.env.distributions, OT_column, self.env.opening_time, increase_surgery_duration=self.env.increase_surgery_duration, increase_ward_stay=self.env.increase_ward_stay)
                        for j in patient_info:
                            create_patient(j, OT_column)
                            self.env.total_number_of_surgeries += 1

                    else:  # If there is more than one specialty that takes place in that OT
                        start_time = self.env.opening_time
                        for p, cell in enumerate(current_cell):
                            ending_time, patient_info = assign_surgery(cell, self.env.distributions, OT_column, start_time, p, increase_surgery_duration=self.env.increase_surgery_duration, increase_ward_stay=self.env.increase_ward_stay)
                            for j in patient_info:
                                create_patient(j, OT_column, second_category=(p > 0))
                                self.env.total_number_of_surgeries += 1
                            start_time = ending_time

            # Hold for 1440 minutes (one day)
            self.hold(1440)
            self.env.days += 1

class patients(sim.Component):
    """
    A class to represent a patient flow in the simulation environment.This class is responsible for setting up the
    patient's surgery details and processing their journey through the simulation.

    Methods:
    - setup(): Sets up the patient details.
    - process(): Main process that the patient will go through in the simulation.
    """

    def setup(self, category, group, OT, surgery_duration, start_time, arrival_time, average_LOS, average_duration_surgery, total_ward_stay, second_category=False):
        """
        Set up the patient details.
        """
        self.category = category
        self.group = group
        self.OT = OT
        self.surgery_duration = surgery_duration
        self.start_time = start_time
        self.arrival_time = arrival_time
        self.average_LOS = average_LOS
        self.average_duration_surgery = average_duration_surgery
        self.total_ward_stay = total_ward_stay
        self.second_category = second_category

        # Calculate the probability of needing ICK and determine if the patient will require ICK
        IC_probability = (self.env.distributions.loc[self.env.distributions['Group'] == self.group, 'Probability_IC_Assignment'].values[0]) / 100
        IC_probability = min(IC_probability * self.env.increase_IC_probability, 1)
        self.IC = np.random.RandomState().choice([True, False], p=[IC_probability, 1 - IC_probability])

    def process(self):
        """
        Main process that the patient will go through in the simulation.This process handles the patient's arrival,
        surgery, potential ICK stay, and ward stay. It saves the information of the patient in the designated lists and dataframes
        """
        while True:
            # Determine the current day in the 28-day cycle and the simulation day
            schedule_day = self.env.days % 28
            simulation_surgery_day = self.env.days

            # Hold the patient until their arrival time
            self.hold(self.arrival_time)

            # Calculate the maximum waiting time for the surgery
            maximum_waiting_time = max(0, self.env.close_time - (self.env.now() % 1440) - self.average_duration_surgery - self.env.cleaning_time)

            # Assign a ward to the patient
            self.ward = assigning_ward(self.env, self.average_LOS, self.category, self.arrival_time)

            # If the patient requires ICK, handle ICK assignment
            if self.IC:
                Assigned_ICK = Assining_ICK(self.category)
                leavingtime = self.env.now() + 60 + self.surgery_duration + self.total_ward_stay
                if Assigned_ICK == 'ICK2_3':
                    if len(self.env.IC_list_2_3) < self.env.resource_wards['ICK2_3'].capacity():
                        self.env.IC_list_2_3.append(leavingtime)
                    elif len(self.env.IC_list_2_3) != 0 and min(self.env.IC_list_2_3) < (self.env.now() + 60 + self.average_duration_surgery):
                        min_index = self.env.IC_list_2_3.index(min(self.env.IC_list_2_3))
                        self.env.IC_list_2_3[min_index] = leavingtime
                    else:
                        self.env.failed_ward_unavailable.append({'ward': 'ICK2_3', 'schedule_day': schedule_day})
                        break
                elif Assigned_ICK == 'ICK1_4':
                    if len(self.env.IC_list_1_4) < self.env.resource_wards['ICK1_4'].capacity():
                        self.env.IC_list_1_4.append(leavingtime)
                    elif len(self.env.IC_list_1_4) != 0 and min(self.env.IC_list_1_4) < (self.env.now() + 60 + self.average_duration_surgery):
                        min_index = self.env.IC_list_1_4.index(min(self.env.IC_list_1_4))
                        self.env.IC_list_1_4[min_index] = leavingtime
                    else:
                        self.env.failed_ward_unavailable.append({'ward': 'ICK1_4', 'schedule_day': schedule_day})
                        break

            # Request the assigned ward resource
            if isinstance(self.ward, list):
                self.request(self.env.resource_wards[self.ward[0]], self.env.resource_wards[self.ward[1]], priority=1, fail_delay=maximum_waiting_time)
            else:
                self.request(self.env.resource_wards[self.ward], priority=1, fail_delay=maximum_waiting_time)

            # Check if the request failed
            if self.failed():
                self.env.failed_ward_unavailable.append({'ward': self.ward, 'schedule_day': schedule_day})
            else:
                # Calculate the maximum waiting time for the OT
                maximum_waiting_time = self.env.close_time - (self.env.now() % 1440) - self.average_duration_surgery - self.env.cleaning_time
                if self.second_category:
                    maximum_waiting_time -= self.env.cleaning_time
                maximum_waiting_time = max(maximum_waiting_time, 0)

                # Request the OT resource
                self.request(self.env.resource_OT[str(self.OT)], fail_delay=maximum_waiting_time)

                # Check if the request failed
                if self.failed():
                    self.env.failed_OT_unavailable.append({'OT': self.OT, 'schedule_day': schedule_day})
                    self.release()
                else:
                    # Handle cleaning time for second category patients
                    if self.second_category:
                        self.hold(self.env.cleaning_time)

                    actual_start_time = self.env.now() % 1440

                    # Handle ICK updates if there is a delay
                    delay_time = actual_start_time - self.start_time
                    if delay_time > 0 and self.IC:
                        if Assigned_ICK == 'ICK2_3' and leavingtime in self.env.IC_list_2_3:
                            index_of_time = self.env.IC_list_2_3.index(leavingtime)
                            leavingtime += delay_time
                            self.env.IC_list_2_3[index_of_time] = leavingtime
                        elif Assigned_ICK == 'ICK1_4' and leavingtime in self.env.IC_list_1_4:
                            index_of_time = self.env.IC_list_1_4.index(leavingtime)
                            leavingtime += delay_time
                            self.env.IC_list_1_4[index_of_time] = leavingtime

                    # Hold for surgery duration and cleaning time
                    self.hold(self.surgery_duration)
                    self.hold(self.env.cleaning_time)
                    self.release(self.env.resource_OT[str(self.OT)])

                    # Check for OT overtime
                    if self.env.now() % 1440 >= 930:
                        self.env.OT_into_overtime.append({'OT': self.OT, 'overtime': (self.env.now() % 1440) - 930, 'schedule_day': schedule_day})

                    # Calculate remaining stay in the ward
                    stay_left = max(self.total_ward_stay - self.surgery_duration - min(self.arrival_time - actual_start_time, 60), 0)
                    time_till_closing_daycare = 1080 - (self.env.now() % 1440)

                    # Handle day care ward closing time
                    if self.ward == 'Daycare' and (stay_left - time_till_closing_daycare) < 60 and stay_left > 0:
                        new_ward = assigning_ward(self.env, self.average_LOS, self.category, self.arrival_time, no_daycare=1)
                        if isinstance(new_ward, list):
                            self.env.resource_wards[new_ward[0]].set_capacity(self.env.resource_wards[new_ward[0]].capacity() + 1)
                            self.request(self.env.resource_wards[new_ward[0]], priority=1)
                            self.env.resource_wards[new_ward[0]].set_capacity(self.env.resource_wards[new_ward[0]].capacity() - 1)
                            self.env.overcapcitation_count += 1
                        else:
                            self.request(self.env.resource_wards[new_ward], priority=1)
                    elif self.IC:
                        self.release()
                        if self.env.resource_wards[Assigned_ICK].available_quantity() > 0:
                            self.request(self.env.resource_wards[Assigned_ICK], priority=1)
                        else:
                            self.env.resource_wards[Assigned_ICK].set_capacity(self.env.resource_wards[Assigned_ICK].capacity() + 1)
                            self.request(self.env.resource_wards[Assigned_ICK], priority=1)
                            self.env.resource_wards[Assigned_ICK].set_capacity(self.env.resource_wards[Assigned_ICK].capacity() - 1)
                            self.env.overcapcitation_count += 1

                    # Hold for the remaining stay
                    self.hold(stay_left)

                    # Ensure patients are not discharged between 11 PM and 7:30 AM
                    if (self.env.now() % 1440) < 450:
                        self.hold(450 - (self.env.now() % 1440))
                    elif (self.env.now() % 1440) > 1380:
                        self.hold(1440 - (self.env.now() % 1440) + 450)

                    self.release()

                    # Remove patient from ICK list if applicable
                    if self.IC:
                        if Assigned_ICK == 'ICK2_3' and leavingtime in self.env.IC_list_2_3:
                            self.env.IC_list_2_3.remove(leavingtime)
                        elif Assigned_ICK == 'ICK1_4' and leavingtime in self.env.IC_list_1_4:
                            self.env.IC_list_1_4.remove(leavingtime)

                    # Record successful surgery
                    self.env.succesfull.append({'OT': self.OT, 'surgery_duration': self.surgery_duration, 'schedule_day': schedule_day, 'simulation_day': simulation_surgery_day})

            break

def run_simulation(seed, distributions = 'Model_Data/Distributie_parameters.xlsx',  dataframe_schedule_surgery = 'Model_Data/Schedule.xlsx',ward_capacity = 'low', model_version = 1, schedule_repeats = 1, lower_ward_capacity = 1, increase_surgery_duration =1, increase_ward_stay = 1, increase_IC_probability =1 ):
    """
    Run the simulation with the provided parameters.

    Parameters:
    - seed (int): Seed for random number generation.
    - distributions (str): Path to the Excel file containing distribution parameters.
    - dataframe_schedule_surgery (str): Path to the Excel file containing the surgery schedule.
    - ward_capacity (str): Ward capacity setting ('low' or 'high').
    - model_version (int): Version of the simulation model.
    - schedule_repeats (int): Number of schedule repeats.
    - lower_ward_capacity (float): Factor to lower the ward capacity.
    - increase_surgery_duration (float): Factor to increase the surgery duration.
    - increase_ward_stay (float): Factor to increase the ward stay duration.
    - increase_IC_probability (float): Factor to increase the intensive care probability.

    Returns:
    - df_failed_ward_unavailable (DataFrame): DataFrame of surgeries failed due to ward unavailability.
    - df_failed_OT_unavailable (DataFrame): DataFrame of surgeries failed due to OT unavailability.
    - df_succesfull (DataFrame): DataFrame of successful surgeries.
    - df_resource_use (DataFrame): DataFrame of resource usage.
    - df_OT_into_overtime (DataFrame): DataFrame of OT into overtime occurrences.
    - total_number_of_surgeries (int): Total number of surgeries.
    - overcapcitation_count (int): Number of times a ward had to overcapacitate.
    """
    # Set the seed for random number generation
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize the simulation environment
    env = sim.Environment(time_unit = 'minutes', trace = False, random_seed = seed)
    env.days = 0 #settting day count to zero
    env.opening_time = 480 # Opening time of the OT in minutes
    env.close_time = 975 # Closing time of the OT in minutes (including potentail overtime of 45 min)
    env.cleaning_time = 15 # Cleaning time between surgeries in minutes
    model_version = str(model_version) #setting the schedule version
    env.distributions = pd.read_excel(distributions) # Load distribution parameters
    df_schedule_surgery = pd.read_excel(dataframe_schedule_surgery, sheet_name=model_version)
    df_schedule_surgery = df_schedule_surgery.map(load_json)
    df_schedule_surgery = setup_schedule_repeats(df_schedule_surgery, schedule_repeats)

    # Variables to adjust model behavior and uncertainty testing
    env.lower_ward_capacity = lower_ward_capacity
    env.increase_surgery_duration = increase_surgery_duration
    env.increase_ward_stay = increase_ward_stay
    env.increase_IC_probability = increase_IC_probability

    # Initialize ward resources
    env.resource_wards = {}
    df_ward_multiplied = df_ward * lower_ward_capacity
    if ward_capacity == 'low':
        env.df_ward_schedule= np.floor(df_ward_multiplied).astype(int)
    elif ward_capacity == 'high':
        env.df_ward_schedule = np.ceil(df_ward_multiplied).astype(int)
    env.df_ward_schedule = setup_schedule_repeats(env.df_ward_schedule, schedule_repeats*4)
    env.column_names_wards = env.df_ward_schedule.columns.tolist()
    for i in env.column_names_wards[2:]:
        env.resource_wards[i] = sim.Resource(i, capacity = env.df_ward_schedule[i].iloc[0])

    # Initialize OT resources
    env.resource_OT = {}
    env.df_OT_availability = setup_schedule_repeats(df_OT, schedule_repeats * 4)
    env.column_names_OT = env.df_OT_availability.columns.tolist()
    for i in env.column_names_OT[2:]:
        env.resource_OT[i] = sim.Resource(i, capacity = env.df_OT_availability[i].iloc[0])

    # Initialize lists to store simulation data
    env.failed_ward_unavailable = []
    env.failed_OT_unavailable = []
    env.succesfull = []
    env.overcapcitation_count = 0
    env.OT_into_overtime = []
    env.IC_list_2_3 = [] #a list to store the IC use for patients that need to be vented
    env.IC_list_1_4 = [] #a list to store the IC use for tpatients that do not need to be vented
    env.total_number_of_surgeries = 0

    # Run the simulation
    TimestepAction(urgent = True)
    day_step(df_schedule_surgery=df_schedule_surgery, urgent = False)
    env.run(till=((schedule_repeats*28*1440)-1))

    # Convert collected data to DataFrames
    df_resource_use = pd.DataFrame(env.resource_data)
    df_failed_ward_unavailable = pd.DataFrame(env.failed_ward_unavailable)
    df_failed_OT_unavailable= pd.DataFrame(env.failed_OT_unavailable)
    df_succesfull = pd.DataFrame(env.succesfull)
    df_OT_into_overtime = pd.DataFrame(env.OT_into_overtime)

    # Return the results
    return df_failed_ward_unavailable, df_failed_OT_unavailable, df_succesfull, df_resource_use, df_OT_into_overtime, env.total_number_of_surgeries, env.overcapcitation_count