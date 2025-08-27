#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
import itertools
import joblib
from tabulate import tabulate
import os
import boto3
import json
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import yaml
from sagemaker.sklearn.estimator import SKLearn
from pathlib import Path
from timeutils import Stopwatch
import logging
import argparse
import tarfile
import warnings
import optuna
# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)



# In[3]:


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, label encoding, and dropping unnecessary columns.
    """
    # Handling missing values and removing duplicate rows
    total = df.shape[0]
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        per = (null_count / total) * 100
        print(f"{col}: {null_count} ({round(per, 3)}%)")

    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Initialize label encoders for categorical features
    encoding_dict = {"protocol_type": LabelEncoder(), "service": LabelEncoder(), "flag": LabelEncoder()}

    # Convert categorical columns to numeric using label encoding
    categorical_columns = ['protocol_type', 'service', 'flag']
    for column in categorical_columns:
        df[column] = df[column].astype('category').cat.codes  # Convert to category codes

    # Perform label encoding on any remaining object columns
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

    # Drop unnecessary columns
    if 'num_outbound_cmds' in df.columns:
        df.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # Save the encoders to a file
    joblib.dump(encoding_dict, 'encoders.joblib')

    return df


def feature_selection(df, target_column='class'):
    """
    Perform feature selection to separate features and target variable.
    
    Args:
    df (pd.DataFrame): The DataFrame to process.
    target_column (str): The name of the target column.

    Returns:
    tuple: A tuple containing the features (X) and labels (y) if target_column exists.
    """
    if target_column in df.columns:
        X = df.drop([target_column], axis=1)  # Features
        y = df[target_column]  # Target variable
        return X, y
    else:
        X = df  # Features
        return X, None  # No target variable


def scale_and_split_data(X, y):
    """
    Scale the data and split it into training and testing sets.
    """
    # Initialize the scaler
    scale = StandardScaler()
    
    # Scale the features
    X_scaled = scale.fit_transform(X)
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.70, random_state=2)
    
    return x_train, x_test, y_train, y_test


def get_data(file_path, target_column=None):
    """
    Load data from a CSV file and split into features (X) and labels (y).

    Args:
    file_path (str): The full path to the CSV file, can be a local path or S3 URI.
    target_column (str): The name of the target column (for training data).

    Returns:
    tuple: A tuple containing the features (X) and labels (y) if target_column exists.
    """
    print(f"Attempting to read file: {file_path}")
    data = pd.read_csv(file_path)
    
    # Preprocess the data
    data = preprocess_data(data)
    
    if target_column and target_column in data.columns:
        X, y = feature_selection(data, target_column)
    else:
        X, y = feature_selection(data)  # No target column in test data
    
    return X, y



def train_and_evaluate_lgbm(x_train, y_train, x_test, y_test, random_state=42):
    """
    Trains a LightGBM model and evaluates it on training and test data.

    Args:
    x_train (pd.DataFrame or np.array): Features for training.
    y_train (pd.Series or np.array): Labels for training.
    x_test (pd.DataFrame or np.array): Features for testing.
    y_test (pd.Series or np.array): Labels for testing.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: A tuple containing the trained model and a dictionary with the training and test scores.
    """
    # Initialize the model
    lgb_model = LGBMClassifier(random_state=random_state)
    
    # Train the model
    lgb_model.fit(x_train, y_train)
    
    # Evaluate the model
    lgb_train_score = lgb_model.score(x_train, y_train)
    
    scores = {"Training Score": lgb_train_score}
    
    if y_test is not None:
        lgb_test_score = lgb_model.score(x_test, y_test)
        scores["Test Score"] = lgb_test_score
    
    # Print the results
    print(f"Training Score: {lgb_train_score}")
    if y_test is not None:
        print(f"Test Score: {lgb_test_score}")
    
    # Return the model and scores in a tuple
    return lgb_model, scores

if __name__ == '__main__':
    print("Starting training job...")
    logger.debug("Starting training job...")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--random_state', type=int, default=42)
    
    # Data directories (can be local or S3 paths)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # Model directory: SM_MODEL_DIR is always set to /opt/ml/model
    parser.add_argument('--sm_model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--gpu-count', type=int, default=os.environ.get('SM_NUM_GPUS'))
    
    args, _ = parser.parse_known_args()
    print("Loading data...")
    logger.debug("Loading data...")
    
    # Load the training data (including target variable)
    train_file = os.path.join(args.train, 'Train_data.csv')
    x_train, y_train = get_data(train_file, target_column='class')
    
    # Load the test data (without target variable)
    test_file = os.path.join(args.test, 'Test_data.csv')
    x_test, _ = get_data(test_file)  # Test data won't have target variable
    
    if y_train is None:
        raise ValueError("Training data must contain the target column.")
    
    # Train and evaluate the model
    print("Training model...")
    logger.debug("Training model...")
    model, scores = train_and_evaluate_lgbm(x_train, y_train, x_test, None, random_state=args.random_state)
    
    print(f"Training Score: {scores['Training Score']}", end='')
    if "Test Score" in scores:
        print(f", Test Score: {scores['Test Score']}")
    else:
        print()  # If no test score available, print a new line
    
    logger.debug(f"Training Score: {scores['Training Score']}", end='')
    if "Test Score" in scores:
        logger.debug(f", Test Score: {scores['Test Score']}")
    else:
        logger.debug("")
    
    # Save the model
    print("Saving model...")
    logger.debug("Saving model...")

    # # 2 Save the model locally
    # local_model_path = os.path.join('/opt/ml/model', 'model.joblib')
    # joblib.dump(model, local_model_path)
    
     joblib.dump(model, os.path.join(args.sm_model_dir, 'model.joblib'))
    # Save the model to the SageMaker model directory or a local path
    
    # #3 Check if running in a SageMaker environment
    # if args.sm_model_dir.startswith("/opt/ml/"):
    #     # Use SageMaker-provided model directory
    #     model_path = os.path.join(args.sm_model_dir, 'model.joblib')
    # else:
    #     # Running locally, use a local path
    #     if not os.path.exists(args.sm_model_dir):
    #         os.makedirs(args.sm_model_dir)
    #     model_path = os.path.join(args.sm_model_dir, 'model.joblib')
    
    # # Save the model
    # joblib.dump(model, model_path)
    
    # print(f"Model saved at {model_path}.")
    # logger.debug(f"Model saved at {model_path}.")
    
    # #4 Local model directory
    # local_model_dir = '/opt/ml/model'
    # local_model_path = os.path.join(local_model_dir, 'model.joblib')
    
    # os.makedirs(local_model_dir, exist_ok=True)
    # joblib.dump(model, local_model_path)
    
    # # Upload to S3
    # s3_client = boto3.client('s3')
    # s3_bucket = args.sm_model_dir.split('/')[2]
    # s3_prefix = '/'.join(args.sm_model_dir.split('/')[3:])
    # s3_model_path = os.path.join(s3_prefix, 'model.joblib')
    
    # print("Uploading model to S3...")
    # s3_client.upload_file(local_model_path, s3_bucket, s3_model_path)
    
    # print(f"Model successfully uploaded to: s3://{s3_bucket}/{s3_model_path}")
    # logger.debug(f"Model successfully uploaded to: s3://{s3_bucket}/{s3_model_path}")    

    # #5 Define local model directory
    # local_model_dir = '/opt/ml/model'  # SageMaker's default model directory
    # local_model_path = os.path.join(local_model_dir, 'model.joblib')
    
    # # Save the model locally
    # os.makedirs(local_model_dir, exist_ok=True)  # Ensure the local directory exists
    # joblib.dump(model, local_model_path)
    
    # # Compress the model into a tar file
    # tar_path = os.path.join(local_model_dir, 'model.tar.gz')
    
    # with tarfile.open(tar_path, "w:gz") as tar:
    #     tar.add(local_model_path, arcname='model.joblib')
    
    # # Upload the tar file to S3
    # s3_client = boto3.client('s3')
    
    # # Extract the bucket and prefix from the S3 URI
    # s3_bucket = args.sm_model_dir.split('/')[2]
    # s3_prefix = '/'.join(args.sm_model_dir.split('/')[3:])
    
    # # Define the final S3 path for the model tar file
    # s3_model_tar_path = os.path.join(s3_prefix, 'model.tar.gz')
    
    # # Upload the tar file to S3
    # print("Uploading model tar file to S3...")
    # s3_client.upload_file(tar_path, s3_bucket, s3_model_tar_path)
    
    # print(f"Model tar file successfully uploaded to: s3://{s3_bucket}/{s3_model_tar_path}")
    
    # print("Model saved.")
    # logger.debug("Model saved.")


# In[ ]:




