# Disaster Response Pipeline Project

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.




app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md
READ_ME.md



### Instructions:


0. pip install -U scikit-learn
1. 
# this is for updating scikit learn packages otherwise some functions will not work

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directcdory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
