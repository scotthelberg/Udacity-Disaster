import sys
import pandas as pd
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loading and merging relevant datasets.'''
    #Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories,on = 'id')
    return df


def clean_data(df):
    '''Cleaning data for more suitable analysis. Breaking categories into individual columns.'''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').apply(lambda x:x[0]) 
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1]) 
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.related.replace("2","1",inplace=True)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///'+database_filename)
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, engine, index=False,if_exists='replace')
    
    pass  


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
