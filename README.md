# Disaster Response Pipeline Project - Udacity DSND Project 2

![Intro Pic](pics/intro.png)

## Table of Contents
* [Project Motivation](#motivation)
* [Installing](#installation)
* [Authors](#authors)
* [License](#license)
* [Acknowledgement](#acknowledgement)

<a name="motivation"></a>
## Project Motivation
This project is to analyze disaster data from Figure Eight and build a model for an API that classifies disaster messages.

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

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
