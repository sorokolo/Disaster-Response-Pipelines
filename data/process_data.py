# Importing all libraries
import sys
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    messages_filepath : csv file containing the messages
    categories_filepath :csv file containing the categories
    
    '''
    # Loading the data using pd.read_csv     
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging the datasets  
    df = pd.merge(messages,categories)

    # Returning the dataframes     
    return df


def clean_data(df):
    '''
    df : the daframe you want to apply to cleaning to
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';' , expand = True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = []
    for val in row:
        category_colnames.append(val.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    
    # drop columns that are no longer needed
    df = df.drop('categories',axis = 1)
  
    
    # concatenate the original dataframe with the new `categories` dataframe
    categories['id']= df['id']
    df = pd.concat([df,categories],axis=1)

    # Check if there are duplicates and remove them
    # check number of duplicates
    if(sum(df.duplicated(subset=None, keep='first') != 0)):
        df.drop_duplicates(subset = None,keep = False, inplace = True)
        
        
    return df

def save_data(df, database_filename):
    '''
    df : the dataframe you want to save
    database_filename : the path of the database you want yo save the dataframe to
    
    '''
    print(database_filename)
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = os.path.basename(database_filename).replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()