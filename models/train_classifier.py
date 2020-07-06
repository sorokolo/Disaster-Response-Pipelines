# Import the important libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    database_filepath : the path of the database you want to load the data from
    '''
    
    # Creating an engine and loading the file     
    print(database_filepath)
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    df = df.drop(['child_alone'],axis=1)
    
   
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    #print(X)
    #print(y.columns)
    columns = Y.columns # This will be used for visualization purpose
    
    return X , Y, columns

def tokenize(text):
    '''
    text : the text you want to tokenize
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace url in case we find some     
    detected_urls = re.findall(url_regex, text)
    
    # Tokenize the text     
    tokens = word_tokenize(text)
    
    #lemmatize and normalize the data     
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    classifier = RandomForestClassifier(min_samples_split = 100,min_samples_leaf = 20, max_depth = 8,
                                       max_features = 'sqrt', random_state = 1)
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(classifier)),
    ])
    
    params = {
    
            'clf__estimator__n_estimators': [100, 200],
                                                        }
    cv = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=-1)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    model : the model you want to evaluate
    X_test : the features of your testing set
    Y_test : the labels of your testing set
    category_names : the list of categories 
    '''
    # Predicting for the test set
    y_pred = model.predict(X_test)

    # Calculating the accuracy per classes

    print(classification_report(Y_test.values, y_pred, target_names = category_names))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()