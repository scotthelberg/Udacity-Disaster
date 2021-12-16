import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import multioutput
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import os
def load_data(database_filepath):
    '''Loading data from database.'''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name,con=engine)
    
    X = df ['message'].values
    y = df.iloc[:,4:]
    y=y.astype(int)
    category_names = y.columns
    return X, y, category_names
def tokenize(text):
    '''Normalizing text for more useful search.'''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Spliting text into words using NLTK
    words = word_tokenize (text)
    
    #Removing stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    
    #reduce words to their root form
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_words
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
    ])
    
    parameters = {'vect__ngram_range':((1,1), (1,2))}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose = 3)
    return pipeline
def evaluate_model(model, X_test, y_test, category_names):
    
       #   predict classes for X_test
    prediction = model.predict(X_test)
    #   print out model precision, recall and accuracy
    print(classification_report(y_test, prediction, target_names=category_names))
def save_model(model, model_filepath):
    #filename = 'disasters_model.sav'
    pickle.dump(model,open(model_filepath,'wb'))
    pass
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
