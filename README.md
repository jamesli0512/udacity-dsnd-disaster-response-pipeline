# Disaster Response Pipeline Project - Udacity DSND Project 2

![Intro Pic](pics/intro.png)

## Table of Contents
* [Project Motivation](#motivation)
* [Installing](#installation)
* [Files Description](#filedescription)
* [Authors](#authors)
* [License](#license)
* [Acknowledgement](#acknowledgement)
* [Instructions](#instructions)

<a name="motivation"></a>
## Project Motivation
This project is to analyze disaster data from Figure Eight and build a model for an API that classifies disaster messages.

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis. People can utilize this app to parse text messages during real disaster time and send the messages to an appropriate disaster relief agency, so people in difference emergency situations can be assisted accordingly. During the disaster time when there are limited personal and resouce, this app would greatly improve the efficiency of organizations on their hands to make quick decisions and save time.

This project is divided in the following key sections:

1. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
2. Build a machine learning pipeline to train the which can classify text message in various categories
3. Run a web app which can show model results in real time


<a name="installation"></a>
### Installing
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

The libs used in this project including:
* Machine Learning Libraries: `NumPy`,`SciPy`,`Pandas`,`Sciki-Learn`
* Natural Language Process Libraries: `NLTK`
* SQLlite Database Libraqries: `SQLalchemy`
* Model Loading and Saving Library: `Pickle`
* Web App and Data Visualization: `Flask`,`Plotly`

<a name="filedescription"></a>
### Files Description
The file-folder structure of the project.

```
????????? app     
???   ????????? run.py                           # Flask file that runs app
???   ????????? templates   
???       ????????? go.html                      # Classification result page of web app
???       ????????? master.html                  # Main page of web app    
????????? data                   
???   ????????? disaster_categories.csv          # Dataset to process 
???   ????????? disaster_messages.csv            # Dataset to process
???   ????????? process_data.py                  # Data cleaning process script 
????????? models
???   ????????? train_classifier.py              # ML model process script
|   ????????? classifier.pkl                   # Trained ML model
????????? README.md
```

<a name="authors"></a>
## Authors
* [James Li](https://github.com/jamesli0512)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

<a name="instructions"></a>
## Instructions 

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseMessages.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseMessages.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/
