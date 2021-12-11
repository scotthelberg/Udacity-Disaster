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

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///Disasters.db')
    df = pd.read_sql_table('Disasters', engine)
    X = df ['message'].values
    y = df.iloc[:,4:]
    return X, y, category_names


def tokenize(text):
    #normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #split text into words using NLTK
    words = word_tokenize (text)
    
    #remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    #reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #reduce words to their root form
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed_words


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__max_depth': [10, 20],
    'clf__estimator__n_estimators': [75,200]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    
    X_train, X_test, y_train, y_test = train_test_split (X,y)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    i =0
    for col in y_test:
        print('Feature {}:{}'.format(i+1,col))
        print(classification_report(y_test[col],y_pred[:,i]))
        i=i+1
    accuracy = (y_pred == y_test.values).means()
    print('The model accuracy score is {:.3f}'.format(accuracy))
    
    scores(y_test, y_pred)
    pass


def save_model(model, model_filepath):
    filename = 'disasters_model.sav'
    pickle.dump(cv,open(filename,'wb'))
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