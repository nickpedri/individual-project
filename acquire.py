import pandas as pd
import numpy as np


import prepare as prep


def animals():
    # Load data from .csv
    a = pd.read_csv('AustinAnimalWorld.csv')
    a.columns = a.columns.str.lower().str.replace(' ', '_')

    # Group livestock into 'other'
    a.animal_type = np.where(a.animal_type == 'Livestock', 'Other', a.animal_type)

    # Recalculate age in days for animals
    a.age_upon_outcome = a.age_upon_outcome.astype(str)
    days = []
    weeks = []
    months = []
    years = []
    for (listt, name, mult) in [(days, 'day', 1), (weeks, 'week', 7), (months, 'month', 30), (years, 'year', 365)]:
        for row in a.age_upon_outcome:
            if name in row:
                listt.append(abs(int(row[:2].strip()) * mult))
            else:
                listt.append(0)
    a['days'] = np.array(days)
    a['weeks'] = np.array(weeks)
    a['months'] = np.array(months)
    a['years'] = np.array(years)
    age = a.days + a.weeks + a.months + a.years
    a['age'] = age
    a.age = np.where(a.age > 730, 1095, a.age)
    a.age = np.where(a.age < 30, 28, a.age)

    # Determine sex and if animal is neutered or spayed
    a.sex_upon_outcome = a.sex_upon_outcome.fillna('Unknown')
    gender = []
    neut_spay = []
    for row in a.sex_upon_outcome.astype(str):
        if row == 'Unknown':
            gender.append('unk')
            neut_spay.append(False)
        elif row[0] == 'I':
            if row[7] == 'F':
                gender.append('female')
                neut_spay.append(False)
            if row[7] == 'M':
                gender.append('male')
                neut_spay.append(False)
        elif row[0] == 'N':
            gender.append('male')
            neut_spay.append(True)
        elif row[0] == 'S':
            gender.append('female')
            neut_spay.append(True)
    a['gender'] = gender
    a['neut_spay'] = neut_spay

    # Determine outcomes of each animal
    new_outcomes = []
    for row in a.outcome_type:
        if row in ['Adoption', 'Transfer', 'Return to Owner', 'Euthanasia', 'Died']:
            new_outcomes.append(row)
        else:
            new_outcomes.append('Other')
    a.outcome_type = new_outcomes

    # Determine condition of animal
    condition = []
    for row in a.outcome_subtype:
        if row == 'Rabies Risk':
            condition.append('Abnormal')
        elif row == 'Suffering':
            condition.append('Abnormal')
        elif row == 'Aggressive':
            condition.append('Abnormal')
        elif row == 'Medical':
            condition.append('Abnormal')
        elif row == 'Underage':
            condition.append('Abnormal')
        elif row == 'Behavior':
            condition.append('Abnormal')
        elif row == 'Under Investigation':
            condition.append('Abnormal')
        else:
            condition.append('Normal')
    a['condition'] = condition

    # Determine if animal has a name
    mylist = a.name.isna()
    the_list = [not elem for elem in mylist]
    a.name = the_list

    # Determine breed of animal
    mixed = []
    new_breed = []
    for row in a.breed:
        if 'Mix' in row:
            mixed.append(True)
            new_breed.append(row[:-4])
        else:
            mixed.append(False)
            new_breed.append(row)
    a['new_breed'] = new_breed

    breed1 = []
    breed2 = []
    for breed, mix in zip(a.new_breed, mixed):
        if '/' in breed:
            br = breed.split('/')
            breed1.append(br[0])
            breed2.append(br[1])
        else:
            if mix is True:
                breed1.append(breed)
                breed2.append('mix')
            else:
                breed1.append(breed)
                breed2.append('purebred')
    a['breed1'] = breed1
    a['breed2'] = breed2
    b2_filter = a.breed2.value_counts().head(25).index
    bre2 = []
    for breed in breed2:
        if breed not in b2_filter:
            bre2.append('Other')
        else:
            bre2.append(breed)
    a.breed2 = bre2
    b1_filter = a.breed1.value_counts().head(30).index
    bre1 = []
    for breed in breed1:
        if breed not in b1_filter:
            bre1.append('Other')
        else:
            bre1.append(breed)
    a.breed1 = bre1

    # Filter the colors
    colors = []
    top_colors = a.color.value_counts().head(20).index
    for color in a.color:
        if color in top_colors:
            colors.append(color)
        else:
            colors.append('Other')
    a.color = colors

    # Drop the rest of useless columns
    a = a.drop(columns=['color', 'days', 'weeks', 'months', 'years', 'age_upon_outcome', 'datetime', 'animal_id',
                        'date_of_birth', 'monthyear', 'breed', 'new_breed', 'sex_upon_outcome', 'outcome_subtype'])

    return a
