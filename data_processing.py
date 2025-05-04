import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load financial profile data from a CSV file
    
    Parameters:
    ----------
    file_path : str or file object
        Path to the CSV file or uploaded file object
    
    Returns:
    -------
    pandas.DataFrame
        Loaded data
    """
    # Load the data
    try:
        data = pd.read_csv(file_path)
        
        # Basic validation of the dataset structure
        required_columns = ['budget', 'risk_tolerance', 'investment_horizon', 'investment_type', 'fitness']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
        
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(data):
    """
    Preprocess the financial profile data for ML modeling
    
    Parameters:
    ----------
    data : pandas.DataFrame
        Raw financial profile data
    
    Returns:
    -------
    pandas.DataFrame
        Preprocessed data ready for model training
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill missing numeric values with median
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numeric
    # One-hot encode categorical columns except the target variable
    categorical_cols_to_encode = [col for col in categorical_columns if col != 'fitness']
    
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    # Scale numeric features
    features = [col for col in df.columns if col != 'fitness']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Ensure target variable is binary (0 or 1)
    if 'fitness' in df.columns:
        # If fitness is categorical (e.g., 'Fit'/'Unfit'), convert to binary
        if df['fitness'].dtype == 'object':
            fitness_mapping = {'Fit': 1, 'Unfit': 0}
            df['fitness'] = df['fitness'].map(fitness_mapping)
        
        # Ensure fitness is either 0 or 1
        df['fitness'] = df['fitness'].astype(int)
    
    return df
