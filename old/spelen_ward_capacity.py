import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 2000)
import warnings
import numpy as np
# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

Ward = pd.read_pickle('Input_Data/Clean_data_Ward_list')
#Per ID and surgery there can be multiple entries for the ward column so when putting this in a datat frame multiple lists have to be split up
df_Ward = pd.DataFrame(columns = ['ID', 'Info'])
dubble_list = []
for row in Ward:
    if len(row) == 2: #if ther is one we can add it to the new data frame
        New_line = pd.DataFrame([[row[0], row[1]]], columns = ['ID', 'Info'])
        df_Ward = pd.concat([df_Ward,New_line])
    else:
        if len(row)>2:
            ID = row[0]
            dubble_list.append(ID)
            number_of_entries = len(row)-1
            for i in range(1,number_of_entries):
                New_line = pd.DataFrame([[row[0], row[i]]], columns=['ID', 'Info'])
                df_Ward = pd.concat([df_Ward, New_line])
df_Ward[['Ward', 'Entry_Timestamp', 'Exit_Timestamp', 'Total_Time_Ward']] = df_Ward['Info'].apply(pd.Series)
df_Ward.drop('Info', axis=1, inplace=True)
#splitting date and time as differen varaibles

df_Ward.drop(columns=['Total_Time_Ward', 'ID' ], inplace=True)
print (df_Ward)
# Generate a list of all unique time steps within the range of the data
time_range = pd.date_range(start=df_Ward['Entry_Timestamp'].min(), end=df_Ward['Exit_Timestamp'].max(), freq='1h')
print(time_range)
# Create a DataFrame to track occupancy status for each ward at each time step
wards = df_Ward['Ward'].unique()
occupancy_df = pd.DataFrame(index=time_range)

df_Ward.loc[df_Ward['Ward'] == 'SP4', 'Ward'] = 'SK4'

# Populate the occupancy DataFrame based on entry and exit times
for ward in wards:
    occupancy_df[ward] = None
    ward_entries = df_Ward[df_Ward['Ward'] == ward]
    for entry in time_range:
        count = len(df_Ward[(df_Ward['Ward'] == ward) & (df_Ward['Entry_Timestamp'] <= entry)& (df_Ward['Exit_Timestamp'] >= entry)])
        occupancy_df.loc[entry, ward] = count

# Extract date and create a new column
occupancy_df['date'] = occupancy_df.index.date

# Assign a unique counter for each date
occupancy_df['day_counter'] = occupancy_df.groupby('date').ngroup() + 1

# Drop the 'date' column if no longer needed
occupancy_df.drop(columns=['date'], inplace=True)

occupancy_df['day_counter'] = (occupancy_df['day_counter'] - 1) % 7 + 1


# Display the first few rows of the occupancy dataframe
print(occupancy_df)
occupancy_df = occupancy_df.replace(0,np.NaN)
# Calculate statistics, ignoring NaNs
min_occupancy = occupancy_df.min(skipna=True)
max_occupancy = occupancy_df.max(skipna=True)
mean_occupancy = occupancy_df.mean(skipna=True)
median_occupancy = occupancy_df.median(skipna=True)

# Create a summary DataFrame to display the results
summary_stats_df = pd.DataFrame({
    'Min Occupancy': min_occupancy,
    'Max Occupancy': max_occupancy,
    'Mean Occupancy': mean_occupancy,
    'Median Occupancy': median_occupancy
})

print(summary_stats_df)
occupancy_df.to_excel('Occupancy_weekly.xlsx')
summary_stats_df.to_excel('Occupancy_summary.xlsx')


file_path = 'Occupancy_weekly.xlsx'

occupancy_df = pd.read_excel(file_path, sheet_name='Sheet1')
# Rename the first column to 'timestamp' for clarity
occupancy_df.rename(columns={occupancy_df.columns[0]: 'timestamp'}, inplace=True)

# Convert 'timestamp' column to datetime if not already
occupancy_df['timestamp'] = pd.to_datetime(occupancy_df['timestamp'])

# Drop rows where 'day_counter' is NaN, if any
occupancy_df.dropna(subset=['day_counter'], inplace=True)
df_SB2 = occupancy_df[['SB2', 'timestamp', 'day_counter']].copy()
occupancy_df.drop(columns=['SB2'], inplace=True)

def classify_shift(hour):
    if 7 <= hour < 16:
        return 2
    elif 16 <= hour < 23:
        return 3
    else:
        return 1

occupancy_df['hour'] = df['timestamp'].dt.hour
occupancy_df['shift_counter'] = df['hour'].apply(classify_shift)

def classify_shift_SB2(hour):
    if 7 <= hour < 10:
        return 2
    elif 10 <= hour < 14:
        return 3
    elif 14 <= hour < 18:
        return 4
    else:
        return 1
# Extract the hour from the timestamp
df_SB2['hour'] = df_SB2['timestamp'].dt.hour

# Create a new column to classify each timestamp into the appropriate shift counter
df_SB2['shift_counter'] = df_SB2['hour'].apply(classify_shift_SB2)
print(df_SB2.head())
# Identify ward columns by excluding non-numeric columns
ward_columns = occupancy_df.select_dtypes(include='number').columns

# Group by 'day_counter' and calculate summary statistics for each ward
summary_stats_wards = occupancy_df.groupby(['day_counter', 'shift_counter'])[ward_columns].agg(['mean', 'median', 'var', 'min', 'max'])
summary_stats_wards_df_SB2 = df_SB2.groupby(['day_counter', 'shift_counter'])['SB2'].agg(['mean', 'median', 'var', 'min', 'max'])

print(summary_stats_wards)
output_file_hourly_path = 'summary_stats_wards.xlsx'
summary_stats_wards.to_excel(output_file_hourly_path)

print(summary_stats_wards_df_SB2)
output_file_hourly_path = 'summary_stats_wards_df_SB2.xlsx'
summary_stats_wards_df_SB2.to_excel(output_file_hourly_path)