import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 2000)
import math
from Visualisation import OT_Unavailabilty_visualisation

# Helper function to read all sheets with a specific prefix
def read_dfs_from_excel(file_path, sheet_name_prefix):
    dfs = []
    with pd.ExcelFile(file_path) as xls:
        sheet_names = [sheet for sheet in xls.sheet_names if sheet.startswith(sheet_name_prefix)]
        for sheet in sheet_names:
            dfs.append(pd.read_excel(xls, sheet_name=sheet))
    return dfs
def filter_weekends(df):
    return df[~df['Cycle_day'].isin([5,6,12,13,19,20,26,27])]
# File path of the Excel file
file_path = 'simulation_results.xlsx'

# ImpOTt DataFrames from Excel
# df_failed_ward_unavailable_list = read_dfs_from_excel(file_path, 'Failed_Ward_Unavailable')
# df_failed_OT_unavailable_list = read_dfs_from_excel(file_path, 'Failed_OT_Unavailable')
# df_successful_list = read_dfs_from_excel(file_path, 'Successful')
# df_resource_use_list = read_dfs_from_excel(file_path, 'Resource_Use')
# df_overtime_list = read_dfs_from_excel(file_path, 'Overtime')

# Create a list to store data for each category
