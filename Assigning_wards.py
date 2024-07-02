
def assigning_ward(env, average_ward_stay, category, arrival_time, no_daycare = 0):
    """
        Assign ward based on patient category, the arrival time of the patient and the average time such a patient would
        stay in the hospital. Additionally the function can be used when the patient enters the hospital so assinging a
        day care is possible for short expected stay. The function can also be used when the patient needs to be assinged
        a new ward because the daycare is closing. Then the indicator for daycare is set to 1 and a different unit is assigned.

        Parameters:
        - env (simpy.Environment): Salabim enviroment of the simulation
        - average_ward_stay (int): Average length of ward stay for the patient category.
        - category (str): Category of the patient.
        - arrival_time (int): Time of patient arrival.
        - no_daycare (int, optional): Flag indicating if patient can still be assigned a daycare

        Returns:
        - ward (str or list): Assigned ward, this can be a list of wards based on availability of the wards.
    """
    #defining which ward a patient should be assigned to based on its category
    KCZ_list = ['URO', 'ORTO','ORTR', 'KIC']
    KTC_list = ['CAS', 'LOS']
    KCN_list = [ 'NEU', 'NEC', 'PLCH', 'PLCO', 'KNO', 'KAA']
    day_care_closing = 1080 # Time after which daycare closes (18:00 in minutes)

    if category =='GYN':
        # Gynecology patients go to a specific ward
        return 'SK4SP4'
    elif (arrival_time+average_ward_stay) < day_care_closing and no_daycare == 0:
        # Check if patient can go to daycare based on arrival time and daycare availability
        return 'Daycare'
    elif category in KCZ_list:
        # Category belongs to KCZ list, prioritize KCZ ward
        if env.resource_wards['KCZ'].available_quantity() > 0:
            return 'KCZ'
        elif env.resource_wards['KCN'].available_quantity() > 0:
            return 'KCN'
        else:
            return ['KCZ', 'KCN'] # Both KCZ and KCN are full so patient will wait until first bed becomes available
    elif category in KCN_list:
        if env.resource_wards['KCN'].available_quantity() > 0:
            return 'KCN'
        elif env.resource_wards['KCZ'].available_quantity() > 0:
            return 'KCZ'
        else:
            return ['KCN', 'KCZ'] # Both KCN and KCZ are full so patient will wait until first bed becomes available
    elif category in KTC_list:
        # Category belongs to KTC list, prioritize KTC ward
        if env.resource_wards['KTC'].available_quantity() > 0:
            return 'KTC'
        elif env.resource_wards['MCKG'].available_quantity() > 0:
            return 'MCKG'
        else:
            return ['KTC', 'MCKG'] # Both KTC and MCKG are full so patient will wait until first bed becomes available
    else:
        # The categories that are left prioritize MCKG ward
        if env.resource_wards['MCKG'].available_quantity() > 0:
            return 'MCKG'
        elif env.resource_wards['KTC'].available_quantity() > 0:
            return 'KTC'
        else:
            return ['MCKG', 'KTC'] # Both MCKG and KTC are full so patient will wait until first bed becomes available