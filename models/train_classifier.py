import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Function to load the db table.
    :param: 
            database_filename: Filepath for SQLite database file.
    :return: 
            X: Independent vars.
            y: Dependent vars.
            category_names: list of categories.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponseMessages',engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns.values
    
    return X, y, category_names

def tokenize(text):
    """
    Function to tokenize the text data.
    :param: 
            text: raw text string.
    :return:
            clean_tokens: tokenized text string list.
    """
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize word tokens and remove stop words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

# Custom transformer for extracting starting verb from text
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Class - Starting Verb Extractor:
        This class is to creat a new feature by extracting 
        starting verb from tex.    
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Function to build the sklearn ML pipeline.
    :param: 
            None.
    :return:
            model: sklearn ML pipeline object.
    """
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartingVerbExtractor())
            ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])  
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to report the f1 score, precision and recall 
    for each output category of the dataset.
    :param: 
            model: sklearn ML pipeline object.
            X_test: Test independent vars.
            y_test: Test dependent vars.
            category_names: list of categories.
    :return:
            None.
    """
    # Predict y_pred using trained model
    y_pred = model.predict(X_test)
    
    # Report the f1 score, precision and recall for each output category of the y_pred
    print(classification_report(y_test.values, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Function to export model as a pickle file.
    :param: 
            model: sklearn ML pipeline object.
            model_filepath: Filepath for sklearn ML pipeline object file.
    :return:
            None.
    """
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponseMessages.db classifier.pkl')


if __name__ == '__main__':
    main()