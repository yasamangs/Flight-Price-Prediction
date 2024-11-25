# train.py
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, ParameterGrid
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Define preprocessing function
def preprocess_data(
    df,
    drop_columns=None,
    log_columns=None,
    outlier_removal=False
):
    """

    Args:
    - df (DataFrame): The dataset.
    - target_column (str): Name of the target column.
    - categorical_columns (list): List of categorical columns to encode.
    - drop_columns (list): Columns to drop before training.
    - log_columns (list): List of columns to apply log-transform.
    Returns:
    The cleaned dataframe
    """

    # Apply log transformation to specified columns
    if log_columns:
        for col in log_columns:
            if col in df.columns:
                # log1p to handle zero or near-zero values
                df[col] = np.log1p(df[col])

    # Remove outliers
    if outlier_removal:
        if 'price' in df.columns:
            df = df[df['price'] <= 100000]

        df = df[df['duration'] <= 30]

    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=drop_columns)

    return df


if __name__ == "__main__":

    # Load the dataset
    df = pd.read_csv("flight.csv")

    # EDA insights
    categorical = list(df.dtypes[df.dtypes == 'object'].index)
    categorical.remove('flight')  # Remove unique identifier

    # Preprocess data
    clean_df = preprocess_data(
        df,
        drop_columns=['Unnamed: 0', 'flight'],
        log_columns=['duration', 'price'],
        outlier_removal=True
    )

    # Separate features and target
    X = clean_df.drop(columns=['price'])  # Features
    y = clean_df['price']                 # Target

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert training and validation data into dictionaries for DictVectorizer
    train_dict = X_train.to_dict(orient='records')
    val_dict = X_val.to_dict(orient='records')
    test_dict = X_test.to_dict(orient='records')

    # Apply DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    X_val = dv.transform(val_dict)
    X_test = dv.transform(test_dict)

    # Train and evaluate model
    # Define the specific parameters for the Random Forest model
    rf_params = {
        'n_estimators': 100,         # Default number of trees
        'max_depth': None,           # Allow the trees to grow until all leaves are pure
        'min_samples_split': 5,
        'min_samples_leaf': 1,       # Minimum number of samples required to be at a leaf node
        'random_state': 42           # Ensure reproducibility
    }

    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(**rf_params)

    # Initialize tqdm progress bar
    print("Training the Random Forest Model:")
    for i in tqdm(range(1), desc="Progress"):
        rf_model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = rf_model.predict(X_val)

    # Evaluate the model on the validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Validation R-squared: {val_r2:.4f}")

    # Save dv and model
    with open('rf_model.bin', 'wb') as file:
        pickle.dump((dv, rf_model), file)

    print("Dv and Model saved successfully!")
