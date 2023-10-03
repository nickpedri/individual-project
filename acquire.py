import requests
import pandas as pd
import numpy as np
import os

from env import get_connection

import prepare as prep


def scrape_data(url='link'):
    """ This function is used to pull data from https://swapi.dev/.
        url - Url for the FIRST page of the data."""
    response = requests.get(url)  # Gets response from url
    data = response.json()  # Create json format response
    df = pd.DataFrame(data['results'])  # Create df from results
    page = data.copy()  # Create extra page to 'flip' through
    while page['next'] is None:  # Create a while loop that will continue until last page
        page = requests.get(page['next']).json()
        df = pd.concat([df, pd.DataFrame(page['results'])], ignore_index=True)  # Adds new data to the dataframe
    return df  # Return df


def sql_query(db='None', query='None'):
    """ This is a function to easily and quickly create a SQL query in python.
       db - String name of database which to access
       query - SQL query to run."""
    db_url = get_connection(db)
    df = pd.read_sql(query, db_url)
    return df  # Returns df from the query that was input


def get_opds(dt=True, ind=True, sor=True):
    filename = 'opsd_germany_daily.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = prep.create_index(df, 'date', datetime=dt, index=ind, sort=sor)  # Re-index df with datetime object
        return df  # Returns local file if there is one
    else:
        df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
        df.columns = df.columns.str.lower()  # Lower case column names
        df.columns = df.columns.str.replace('+', '_')  # Replace '+' with '_'
        df = prep.create_index(df, 'date', datetime=dt, index=ind, sort=sor)  # Re-index df with datetime object
        df = df.fillna(0)  # Fill nulls with 0
        df.wind_solar = df.wind + df.solar  # Recreate wind_solar column
        df.to_csv(filename)  # Save to .csv file
        return df  # Return df


def get_items(dt=True, ind=True, sor=True):
    filename = 'items.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename)  # Returns local file if there is one
        df = prep.create_index(df, 'sale_date', datetime=dt, index=ind, sort=sor)  # Re-index df with datetime object
        return df  # Returns local file if there is one
    else:
        query = ''' SELECT sale_date, sale_amount, item_brand, item_name, item_price, 
                    store_address, store_zipcode, store_city, store_state  
                    FROM sales AS s
                    LEFT JOIN items as i USING(item_id)
                    LEFT JOIN stores as st USING(store_id)'''
        df = sql_query('tsa_item_demand', query)
        df = prep.create_index(df, 'sale_date', datetime=dt, index=ind, sort=sor)  # Re-index df with datetime object
        df.to_csv(filename)  # Save to .csv file
        return df  # Return df


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
    a.age = np.where(a.age < 30, 1095, a.age)

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

    # Drop the rest of useless columns
    a = a.drop(columns=['days', 'weeks', 'months', 'years', 'age_upon_outcome', 'datetime', 'animal_id',
                        'date_of_birth', 'monthyear', 'breed', 'new_breed', 'sex_upon_outcome', 'outcome_subtype'])

    return a
