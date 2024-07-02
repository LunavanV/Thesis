import pandas as pd

# Set display options for pandas
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)

# Read data from Excel file
data = pd.read_excel('../Input_Data/Clean_data_OR+Ward.xlsx')

# Rename columns for clarity
data = data.rename(columns={'VerrichtingOfIndicatie': 'Description', 'SpecKOP': 'Category', 'ORtime': 'Duration_Surgery'})

# Filter out rows where 'Category' is 'RON' and create a copy
df_rontgen = data[data['Category'] == 'RON'].copy()

# Remove rows where 'Category' is 'RON' from the original dataset
data = data[data['Category'] != 'RON'].copy()


# Read surgery groups data from another Excel file
df_surgery_groups = pd.read_excel('../Input_data/Surgery_Groups.xlsx')

# Rename columns for consistency
df_surgery_groups = df_surgery_groups.rename(columns={'Naam': 'Description'})

# Merge data with surgery groups data based on 'Description' and 'Category'
df_total = pd.merge(data, df_surgery_groups, on=['Description', 'Category'], how='left', suffixes=('_ward', '_OR'))

# Iterate over rows in df_rontgen to categorize 'Duration_Surgery'
for index, row in df_rontgen.iterrows():
    if row['Duration_Surgery'] <= 76:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'higher'

# Filter rows in df_surgery_groups where 'Category' is 'RON' and categorize 'Group_number'
df_Ron_group = df_surgery_groups[df_surgery_groups['Category'] == 'RON'].copy()
for index, row in df_Ron_group.iterrows():
    if row['Group_number'] == 1:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'higher'

# Merge df_rontgen with df_Ron_group based on 'Description', 'Category', and 'Cat_surgery_duration'
df_rontgen = pd.merge(df_rontgen, df_Ron_group, on=['Description', 'Category', 'Cat_surgery_duration'], how='left')

# Concatenate df_total with df_rontgen to combine the datasets
df_total = pd.concat([df_total, df_rontgen])

# Filter rows in df_total where 'Group_number' is NaN
empty_group_number = df_total[df_total['Group_number'].isna()]

# Create a separate DataFrame with rows where 'Group_number' is NaN
empty_group_number_df = pd.DataFrame(empty_group_number)

# Remove rows from df_total where 'Group_number' is NaN
df_total = df_total.dropna(subset=['Group_number'])

# Create a new column 'Group' based on 'Category' and 'Group_number'
df_total['Group'] = df_total.groupby(['Category', 'Group_number']).ngroup() + 1

# Save df_total to an Excel file without index column
df_total.to_excel('../Input_Data/data_with_group_number.xlsx', index=False)
