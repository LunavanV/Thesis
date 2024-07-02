import pandas as pd
pd.set_option('display.max_columns', 40)

df_surgery_duration = pd.read_excel('../Input_Data/Distributie_parameters_surgery_duration.xlsx', index_col=False)
df_length_of_stay = pd.read_excel('../Input_Data/Distributie_parameters_Length_of_stay.xlsx', index_col=False)
df_IC_probability = pd.read_excel('../Input_Data/IC_probability.xlsx')

# Reset index of df_length_of_stay and df_IC_probability
df_length_of_stay.reset_index(drop=True, inplace=True)
df_IC_probability.reset_index(drop=True, inplace=True)

df_total = pd.merge(df_surgery_duration, df_length_of_stay, on=['Category', 'Group', 'Group_number'], how='inner')
df_total = pd.merge(df_total, df_IC_probability, on=['Category', 'Group_number'], how='inner')

df_total = df_total.drop(columns=['Count 30%', 'Count 100%', 'Total count:'])
df_total = df_total.rename(columns={'Pecentage':'Probability_IC_Assignment'})
print(df_total)

df_total.to_excel('../Model_Data/Distributie_parameters.xlsx')