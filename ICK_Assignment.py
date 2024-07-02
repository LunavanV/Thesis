
def Assining_ICK(category):
    """
        Assign ICK based on patient category. There are three categories that need a ICK with the need for the ability
        to vent the patient. All other are sent to the ICK that can not vent patients.

        Parameters:
        - category (str): Category of the patient.

        Returns:
        - ICU (str): Assigned ICU based on patient category.
    """

    if category == 'CAS' or category == 'NEC'or category == 'KIC':
        # Assign ICU 2-3 for specific categories
        return 'ICK2_3'
    else:
        # Assign ICU 1-4 for all other categories
        return 'ICK1_4'
