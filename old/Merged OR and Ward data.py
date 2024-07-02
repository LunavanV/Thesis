import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)
# Load the data
data = pd.read_excel('Input_Data/Clean_data_OR+Ward.xlsx')

df_surgery_groups = pd.read_excel('Input_Data/Surgery_Groups.xlsx')
df_surgery_groups = df_surgery_groups.rename(columns={'Naam':'Surgery_type', 'Group number' : 'Group_number'})
data = data.rename(columns={'VerrichtingOfIndicatie':'Surgery_type', 'SpecKOP':'Category'})
data = data.rename(columns={})

df_rontgen = data[data['Category'] == 'RON'].copy()
data = data[data['Category'] != 'RON'].copy()

df_total = pd.merge(data, df_surgery_groups, on=['Surgery_type','Category'],how='left', suffixes=('_ward', '_OR'))
for index, row in df_rontgen.iterrows():
    if row['ORtime'] <= 76:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_rontgen.loc[index, 'Cat_surgery_duration'] = 'higher'

df_Ron_group = df_surgery_groups[df_surgery_groups['Category'] == 'RON'].copy()
for index, row in df_Ron_group.iterrows():
    if row['Group_number'] == 1:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'Lower'
    else:
        df_Ron_group.loc[index, 'Cat_surgery_duration'] = 'higher'

df_rontgen = pd.merge(df_rontgen, df_Ron_group, on=['Surgery_type','Category', 'Cat_surgery_duration'],how='left')

df_total = pd.concat([df_total, df_rontgen])

#if still no match can be found na rows are deleted
na_rows = df_total[df_total['Group_number'].isna()]

# Remove the rows with empty 'Group number' from df_total
df_total = df_total.dropna(subset=['Group_number'])
df_total.drop(columns=['Cat_surgery_duration'], inplace=True)

df_total['Group'] = df_total.groupby(['Category', 'Group_number']).ngroup() + 1
# group_data = df_total.groupby('Group').agg({'Specialismecode': 'first', 'Group number': 'first'}).reset_index()
print(df_total)

df_total.to_excel('Input_Data/Merged_data.xlsx', index=False)


