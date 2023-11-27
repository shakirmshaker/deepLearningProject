from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


data_preprocessing = pd.read_csv('../data/raw_data/TelcoCustomerChurn.csv')

# Dropping rows with missing or malformed 'TotalCharges' entries
data_preprocessing['TotalCharges'] = pd.to_numeric(data_preprocessing['TotalCharges'], errors='coerce')
data_preprocessing = data_preprocessing.dropna(subset=['TotalCharges'])

# Define features and target variable
X = data_preprocessing.drop(columns=['Churn', 'customerID'])  
y = data_preprocessing['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X_preprocessed = preprocessor.fit_transform(X)
feature_names = numerical_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(input_features=categorical_features))

# Create a DataFrame with the preprocessed data
data_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
data_preprocessed['Churn'] = y.values

data_preprocessed['CustomerID'] = data_preprocessing['customerID'].values

# Save the preprocessed data to a CSV file
data_preprocessed.to_csv('../data/preprocessed_data/TelcoCustomerChurn_Preprocessed.csv', index=False)