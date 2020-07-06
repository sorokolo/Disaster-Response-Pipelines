# Disaster Response Pipeline Project

##Author 
Soro Kolotioloma

##Overview
A flask web app that run a multi-class classifier on a disaster messages data set provided by Figure eigh

##Files in the repo
This repo contains 4 main folders
1.	App : This repo contains the actual Flask app, it made of a nun.py file and a template folder that contain the htlm of the pages
2.	Data : this folder has all the csv files used , a .py file that cleans the data and a database, were the clean_data is saved
3.	Model : This folder has the .py file that create and run the model and ,pkl file were the model is saved
4.	A README file


##Licence
This project is licensed under the MIT License - see the LICENSE.md file for details

##Acknowledgments
Udacity 

To run the project follow the below instructions:

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

