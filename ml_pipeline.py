import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the processed data into training and testing sets
    
    Parameters:
    ----------
    data : pandas.DataFrame
        Preprocessed financial profile data
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    -------
    X_train, X_test, y_train, y_test : tuple of numpy.ndarray
        Split data for training and testing
    """
    # Separate features and target
    X = data.drop('fitness', axis=1)
    y = data['fitness']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, random_state=42):
    """
    Train a logistic regression model for financial fitness prediction
    
    Parameters:
    ----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    random_state : int, optional (default=42)
        Random seed for reproducibility
    
    Returns:
    -------
    sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    """
    # Initialize the model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data
    
    Parameters:
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    
    Returns:
    -------
    tuple
        (accuracy, precision, recall, f1, confusion_matrix)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix

def predict_fitness(model, financial_profile):
    """
    Predict financial fitness for a given financial profile
    
    Parameters:
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    financial_profile : pandas.Series or pandas.DataFrame
        Financial profile data for prediction
    
    Returns:
    -------
    tuple
        (prediction, probability)
        prediction: 1 for Fit, 0 for Unfit
        probability: probability of being Fit (0-100%)
    """
    # Ensure the input is formatted correctly
    if isinstance(financial_profile, pd.Series):
        profile_data = financial_profile.values.reshape(1, -1)
    else:
        profile_data = financial_profile
    
    # Make prediction
    prediction = model.predict(profile_data)[0]
    
    # Get probability
    probability = model.predict_proba(profile_data)[0][1] * 100
    
    return prediction, probability
