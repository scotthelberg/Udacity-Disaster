# Udacity Disaster Response Analysis

**Libraries Used:**
  - Pandas
  - NLTK
  - re
  - Sklearn

**Files used:**
   - messages.csv - 
   - categories.csv 

**Commands run:**

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

**Project Structure**
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

I am completing this project as part of my requirement in the Data Scientist Nanodegree course at Udacity. I was given the opportunity to analyze messages sent during disasters. This project involves building a ETL pipeline to extract, transform, and load the data. Additionally, I got to develop a machine learning pipeline to classify those messages to make them more useful to relief agencies.

Thank you to the Udacity team for providing the data and giving me exposure to the data science world. Acknowledgements also to RSM for giving me the opportunity to delve into this curiousity and diversify my skillsets. 
