import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load the datasets
    :param: 
            messages_filepath:string: Filepath to messages.csv.
            categories_filepath:string: Filepath to categories.csv.
    :return: 
            df: merged df.
    """
    # Load messages dataset
    messages = pd.read_csv('disaster_messages.csv')
    
    # Load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
    
    # Merge datasets
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Function to clean the merged df
    :param: 
            df: Merged df to be cleaned.
    :return: 
            df: Cleaned ddf.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Extract a list of new column names for categories
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    
    # Replace with 1 for df['related'] > 1 rows
    df['related'].replace(2,1,inplace=True)
    
    return df
    
    
def save_data(df, database_filename):
    """
    Function to clean the merged df
    :param: 
            df: Cleaned df.
            database_filename: Filepath for SQLite database file.
    :return: 
            None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseMessages', engine, index=False, if_exists='replace')


def main():
    """
    Function to load and clean df and store as SQLite db file.
    """
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
              'DisasterResponseMessages.db')


if __name__ == '__main__':
    main()