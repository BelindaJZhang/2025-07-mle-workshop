#!/usr/bin/env python
# coding: utf-8

import datetime
from datetime import date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import argparse 
from loguru import logger 




def read_dataframe(filename):
    """
    Reads a Parquet file into a pandas DataFrame, calculates trip durations,
    filters out unrealistic durations, and prepares categorical columns.

    The function performs the following steps:
    - Loads a Parquet file into a DataFrame.
    - Computes trip duration in minutes based on pickup and dropoff timestamps.
    - Filters out trips with durations less than 1 minute or greater than 60 minutes.
    - Converts pickup and dropoff location IDs to string type for categorical processing.
    Args:
        filename (str): Path to the input Parquet file.
    Returns:
        pandas.DataFrame: Cleaned DataFrame with calculated duration and categorical location columns.
    """
    logger.info(f"loading_file: {filename}")
    try:
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        logger.info(f'{filename} had {len(df)} rows')
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}")
        logger.error(e)
        raise

def train(train_date: date, val_date: date, out_path: str):
    """
        Trains a linear regression model to predict trip duration using taxi trip data,
        and saves the trained pipeline to a file.

        The function:
        - Downloads training and validation data for given dates from a cloud URL.
        - Reads and preprocesses the data using `read_dataframe()`.
        - Extracts categorical and numerical features for modeling.
        - Trains a linear regression model using a pipeline with DictVectorizer.
        - Evaluates the model using root mean squared error (RMSE).
        - Serializes and saves the trained pipeline to a file using pickle.

        Args:
            train_date (date): Date corresponding to the training dataset (used to generate the file URL).
            val_date (date): Date corresponding to the validation dataset (used to generate the file URL).
            out_path (str): File path to save the trained pipeline (as a .pkl file).

        Returns:
            None
        """

    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = base_url.format(year=train_date.year, month=train_date.month)
    val_url = base_url.format(year=val_date.year, month=val_date.month)



    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    target = 'duration'
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    lr = LinearRegression()
    pipeline = make_pipeline(dv, lr)
    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    
    mse = mean_squared_error(y_val, y_pred, squared=False)
    
    logger.info(f'{mse=}') # f'{mse=}' = f'mse={mse}'
    logger.info(f'writing model to {out_path}')


    with open(out_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)

    return mse





