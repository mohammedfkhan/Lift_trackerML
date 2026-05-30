# 🏋️ Weightlifting Strength Milestone Predictor

## Overview
This is a local machine learning pipeline built to forecast strength training progression. It uses a **Random Forest Regressor** to analyze an athlete's current lifting stats and lifestyle metrics to predict exactly how many weeks of consistent training it will take to reach a specific weightlifting milestone.

## Features
* **Custom Data Modeling:** Parses synthetic athlete training data (body weight, current bench, rep volume, sleep score, and consistency days).
* **Predictive Forecasting:** Utilizes an ensemble learning method (Random Forest) to output a customized timeline for strength goals.
* **Accuracy Tracking:** Evaluates model performance using Mean Absolute Error (MAE) to ensure timeline predictions remain realistic.

## Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`, `train_test_split`)

## Quick Start
To run this model locally on your machine:

1. Clone this repository.
2. Create and activate a virtual environment:
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Install the required dependencies:
   `pip install pandas scikit-learn`
4. Run the prediction pipeline:
   `python train_model.py`
