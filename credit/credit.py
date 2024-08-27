import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv(r"E:\german_credit_data.csv")

# Define the target column
target_column = 'Risk'  # Make sure this is the correct target column

# Verify the target column exists
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset")

# Drop rows where the target is NaN
data = data.dropna(subset=[target_column])

# Separate features and target
features = data.drop([target_column, 'Unnamed: 0'], axis=1, errors='ignore')
target = data[target_column]

# Define numerical and categorical columns
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = features.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create the pipeline with RandomForestClassifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Function to predict risk based on user input
def predict_user_input(user_data):
    # Convert the user data to a DataFrame
    user_df = pd.DataFrame([user_data], columns=features.columns)
    
    # Predict the risk using the pipeline
    prediction = pipeline.predict(user_df)
    
    # Return the prediction
    return prediction[0]

# Example user input
user_input = {

    'Age': 65,
    'Sex': 'male',
    'Job': 2,
    'Housing': 'own',
    'Saving accounts': 'little',
    'Checking account': 'little',
    'Credit amount': 571,
    'Duration': 21,
    'Purpose': 'car'
}

# Predict the risk for the example user input
result = predict_user_input(user_input)
print(f"The predicted risk for the user is: {result}")