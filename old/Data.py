import pandas as pd
from datetime import datetime, time, timedelta
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 2000)

#this function is used for combining the entries were there subsequent of one and other and the ward are they same
def combining_subsequent_entries(df,i,length):
    df = df.sort_values(by=['Entry_Date', 'Entry_Time']).reset_index(drop=True)
    if i >= length - 1:  # Termination condition: stop if index i is beyond the last index
        return df

    if df['Exit_Date'].iloc[i] == df['Entry_Date'].iloc[i+1] and df['Ward'].iloc[i] == df['Ward'].iloc[i+1]:
        df.at[i, 'Exit_Date'] = df['Exit_Date'].iloc[i + 1]
        df.at[i, 'Exit_Time'] = df['Exit_Time'].iloc[i + 1]
        df.at[i, 'Total_Time_Ward'] += df['Total_Time_Ward'].iloc[i + 1]
        df = df.drop(df.index[i+1]).reset_index(drop=True)
        # After modifying the DataFrame, return the result of the recursive call
        return combining_subsequent_entries(df, i, len(df))
    else:
        # After modifying the DataFrame, return the result of the recursive call
        return combining_subsequent_entries(df, i + 1, len(df))

#This function is for creating an extra column for different ward stays belonging to the same surgery
def splitting_different_ward_stays(df, df_total):
#If the dataset consist of one line it will simply be added to the total dataframe, this might happen after recursion
    if len(df)>1:
        df = df.sort_values(by=['Entry_Date', 'Entry_Time'])
#looping trough each element of the dataframe except the last one since this will cause an error when comparing it to the next line
        for i in range(len(df)-1):
            if df['Exit_Date'].iloc[i] != df['Entry_Date'].iloc[i+1]: # if the next line exit time is not equal to this lines entry time we will combine this line with all the previous lines (if any) and at the combined line to the dataset
                Line_To_Add = df.iloc[0:1].copy()
                for j in range(1,i+1):
                    Line_To_Add[f'Ward{j+1}'] = df['Ward'].iloc[j]
                    Line_To_Add[f'Ward{j+1}_time'] = df['Total_Time_Ward'].iloc[j]
                df_total = pd.concat([df_total, Line_To_Add])
                for k in range(i+1): #Deleting all the lines we have used for the combined line
                    df.drop(df.index[0], inplace=True)
                df.reset_index(drop=True, inplace=True)
                splitting_different_ward_stays(df, df_total) #recalling the function with the data that remains
                return df_total

            elif i == (len(df)-2): #if we get to the second to last line before satisfying the previous condition we are combining all and adding them to the dataset
                Line_To_Add = df.iloc[0:1].copy()
                for j in range(1, len(df)):
                    Line_To_Add[f'Ward{j + 1}'] = df['Ward'].iloc[j]
                    Line_To_Add[f'Ward{j + 1}_time'] = df['Total_Time_Ward'].iloc[j]
                df_total = pd.concat([df_total, Line_To_Add])
                return df_total

    else: #if there is just one line left we are adding it to the data set
        Line_To_Add = df.iloc[0:1].copy()
        df_total = pd.concat([df_total, Line_To_Add])
        return df_total

#First import both the OR data and the Ward data
OR = pd.read_pickle('Input_Data/Clean_data_OR_list')
Ward = pd.read_pickle('Input_Data/Clean_data_Ward_list')

#Turn the OR data set into a regular dataframe
columns = ['ID', 'info', 'Category'] #create the first columns because the info is a list of information
df_OR = pd.DataFrame(OR, columns=columns)
#Split up the second column of list into different columns and drop the column with the list
df_OR[['Type', 'Start_time_Surgery', 'End_time_Surgery', 'Duration_Surgery', 'Department_Surgery', 'Description']] = df_OR['info'].apply(pd.Series)
df_OR.drop('info', axis=1, inplace = True)

#splitting the date and time variables in the dataframe
df_OR['Start_Date'] = df_OR['Start_time_Surgery'].dt.date
df_OR['Start_Time'] = df_OR['Start_time_Surgery'].dt.time
df_OR.drop(columns=['Start_time_Surgery'], inplace=True)
df_OR['End_Date'] = df_OR['End_time_Surgery'].dt.date
df_OR['End_Time'] = df_OR['End_time_Surgery'].dt.time
df_OR.drop(columns=['End_time_Surgery'], inplace=True)

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
df_Ward[['Ward', 'Timestamp_entry', 'Time_stap_exit', 'Total_Time_Ward']] = df_Ward['Info'].apply(pd.Series)
df_Ward.drop('Info', axis=1, inplace=True)
#splitting date and time as differen varaibles
df_Ward['Entry_Date'] = df_Ward['Timestamp_entry'].dt.date
df_Ward['Entry_Time'] = df_Ward['Timestamp_entry'].dt.time
df_Ward.drop(columns=['Timestamp_entry'], inplace=True)

df_Ward['Exit_Date'] = df_Ward['Time_stap_exit'].dt.date
df_Ward['Exit_Time'] = df_Ward['Time_stap_exit'].dt.time
df_Ward.drop(columns=['Time_stap_exit'], inplace=True)
# df_Ward.to_excel('df_Ward_1.xlsx', index=False)
#Now entering subsequent OR states
grouped = df_Ward.groupby('ID')
df_Ward_Total = pd.DataFrame()
biggest_switches = 0
for group_name, group_data in grouped:
    if group_data.shape[0] > 1:
        shortened_group = combining_subsequent_entries(group_data,0, len(group_data))
        shortened_group = shortened_group.reset_index()
        df_Ward_Total = splitting_different_ward_stays(shortened_group, df_Ward_Total)
    else:
        df_Ward_Total = pd.concat([df_Ward_Total, group_data])
# df_Ward_Total.to_excel('df_Ward_2.xlsx', index=False)
merged_df = pd.merge(df_OR, df_Ward_Total, how='left', left_on=['Start_Date', 'ID'], right_on=['Entry_Date', 'ID'], suffixes=('_OR', '_Ward'))
df_with_ward = merged_df[merged_df['Ward'].notnull()]
df_without_ward = merged_df[merged_df['Ward'].isnull()]

Empty_count = 0
df_without_ward_still = pd.DataFrame()
for index, row in df_without_ward.iterrows():
    current_id = row['ID']
    matching_rows = df_Ward_Total[df_Ward_Total['ID'] == current_id]
    if not matching_rows.empty and row['Start_Date'] <= matching_rows.iloc[0]['Exit_Date']:
        df_row = row.to_frame().transpose()
        df_row = df_row.dropna(axis=1, how='all')
        df_combined_rows = pd.merge(df_row, matching_rows, how= 'left', on='ID')
        df_with_ward = pd.concat([df_with_ward, df_combined_rows])
    else:
        Empty_count += 1
        df_row = row.to_frame().transpose()
        df_without_ward_still = pd.concat([df_without_ward_still, df_row])

#importing the surgery groups based on their name
df_rontgen = df_with_ward[df_with_ward['Category'] == 'RON'].copy()
df_with_ward = df_with_ward[df_with_ward['Category'] != 'RON'].copy()

df_surgery_groups = pd.read_excel('Input_data/Surgery_Groups.xlsx')
df_surgery_groups = df_surgery_groups.rename(columns={'Naam':'Description'})
df_total = pd.merge(df_with_ward, df_surgery_groups, on=['Description','Category'],how='left', suffixes=('_ward', '_OR'))

for index, row in df_rontgen.iterrows():
    if row['Duration_Surgery'] <= 76:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'higher'

df_Ron_group = df_surgery_groups[df_surgery_groups['Category'] == 'RON'].copy()
for index, row in df_Ron_group.iterrows():
    if row['Group number'] == 1:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'higher'

df_rontgen = pd.merge(df_rontgen, df_Ron_group, on=['Description','Category', 'Cat_surgery_duration'],how='left')

df_total = pd.concat([df_total, df_rontgen])

# Filter rows with empty 'Group number'
empty_group_number = df_total[df_total['Group number'].isna()]
# Create a separate DataFrame with these rows
empty_group_number_df = pd.DataFrame(empty_group_number)
# Remove the rows with empty 'Group number' from df_total
df_total = df_total.dropna(subset=['Group number'])

print(f'Two data frames had to be deleted consistend {len(empty_group_number_df)} and {len(df_without_ward_still )} lines of data')

# combining the date time columns again to get a better overview in the dataframe and drop unusefull columns
df_total['Start_date_time'] = pd.to_datetime(df_total['Start_Date'].astype(str) + ' ' + df_total['Start_Time'].astype(str))
df_total['End_date_time'] = pd.to_datetime(df_total['End_Date'].astype(str) + ' ' + df_total['End_Time'].astype(str))
df_total['Entry_date_time'] = pd.to_datetime(df_total['Entry_Date'].astype(str) + ' ' + df_total['Entry_Time'].astype(str))
df_total['Exit_date_time'] = pd.to_datetime(df_total['Exit_Date'].astype(str) + ' ' + df_total['Exit_Time'].astype(str))

df_total['Before_Surgery_Time'] = (df_total['Start_date_time'] - df_total['Entry_date_time']).dt.total_seconds() / 60

df_total.drop(['ID', 'Start_Date', 'Start_Time', 'End_Date','End_Time','Entry_Date','Entry_Time', 'Exit_Date', 'Exit_Time', 'Type', 'index', 'Cat_surgery_duration', 'Type','Start_date_time', 'End_date_time', 'End_date_time', 'Exit_date_time', 'Entry_date_time', 'Department_Surgery'], axis=1, inplace=True)


#calculating the average surgery time for each group, which will be used for the correct scheduling
df_total['Average_Surgery_Time'] = df_total.groupby(['Description', 'Category'])['Duration_Surgery'].transform('mean').round(0)
df_total['Max_Surgery_Time'] = df_total.groupby(['Description', 'Category'])['Duration_Surgery'].transform('max')
df_total['Min_Surgery_Time'] = df_total.groupby(['Description', 'Category'])['Duration_Surgery'].transform('min')


# #Category is now no longer in use since a departent was assigned when the groupings were assinged
# df_total = df_total.drop('Category', axis=1)
print(df_total)
df_total.to_excel('Model_data/df_total.xlsx', index=False)
df_Ward_Total.to_excel('Model_data/df_Ward_Total.xlsx', index=False)
df_OR.to_excel('Model_data/df_OT.xlsx', index=False)

