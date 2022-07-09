''' Required  packages are imported  briefly pandas and numpy for matrix algebra, nltk for  
text process, sqlalchemy for database things, sklearn for ML '''
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
nltk.download('omw-1.4')


from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
# List stop words from NLTK
from nltk.corpus import stopwords

from sqlalchemy import create_engine
import re

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support, make_scorer, accuracy_score, f1_score, fbeta_score, classification_report

import pickle

import sys

# load data  actually reach the database read it and transform it to pandas dataframe

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df =pd.read_sql_table('DisasterResponse',con=engine)   
    X = df.message
    y =df.loc[:, ~df.columns.isin(['id', 'message', 'original', 'genre'])]
    return X, y

# tokenize is a function that process text data like throwing puntuation, lowercase, droping 
#stop words etc.
def tokenize(text):
    # Convert to lowercase
    text=text.lower()
    
    import re
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    

    #Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed

# build model function  builds the ML model using also the tokenize function and asa result it
# return a model to train with our data
def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()), ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # specify parameters for grid search
    parameters = {'clf__estimator__n_estimators': [50, 100]}
    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

# evaluate_model  uses  test part of the data to make predictions and makes an evaluation regarding to 
# the results of precidtions
def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    column_names=y_test.columns
    df_pred=pd.DataFrame(y_pred, columns=column_names)
    for column in y_test:
        print(classification_report(y_test[column], df_pred[column], zero_division=0))
    
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)
    print("Total Accuracy:", accuracy.mean())
    print("\nBest Parameters:", model.best_params_)

# saves the trained model for further uses
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    
# main() is the main function that runs the functions above in a proper way to lead the ML pipeline 
# sequence is  loading data--> spliting to train-test parts --> creating model--> Training Model -->
# --> making predictions and evaluation the results --> saving the model
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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