import math
def set_resource_capacity(env):
    """
        Set resource capacities for wards and operating theaters based on current simulation time.It updates both ward
        and operating theater capacities in the simulation environment over time. Additionally there is a build in way
        to change the capacity in order to perform the sensitivity analysis.

        Parameters:
        - env: Salabim environment object containing resources and schedule data.
        - lower_ward_use (optional): Factor to adjust lower ward capacity (default is 1).

    """

    # Adjust capacities for wards
    for current_resource in env.column_names_wards[2:]:
        current_capacity = env.df_ward_schedule[current_resource].iloc[int(env.now())]
        env.resource_wards[current_resource].set_capacity(current_capacity)

    # Adjust capacities for operating theaters
    for current_resource in env.column_names_OT[2:]:
        current_capacity = env.df_OT_availability[str(current_resource)].iloc[int(env.now())]
        env.resource_OT[str(current_resource)].set_capacity(current_capacity)
