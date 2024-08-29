from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv(r'E:\movi\archive\heart.csv')

# Define target column
target_column = 'HeartDisease'  # Replace with the actual target column name

# Define features and target variable
X = data.drop(target_column, axis=1)
y = data[target_column]

# Ensure no missing values
if X.isnull().sum().any() or y.isnull().sum() > 0:
    raise ValueError("There are missing values in the dataset.")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

# Define the ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

# Define the pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the ROC curve
# Use the pipeline to get predicted probabilities
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()









# Corrected user input with strings in quotes
user_input = {

    'Age': 39,
    'Sex': 'M',
    'ChestPainType': 'ATA',
    'RestingBP': 120,
    'Cholesterol': 204,
    'FastingBS': 0,
    'RestingECG': 'Normal',
    'MaxHR': 145,
    'ExerciseAngina': 'N',
    'Oldpeak': 0,
    'ST_Slope': 'Up'
}

# Define prediction function
def predict_heart_disease(input_data):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Print the DataFrame and its data types
    print("User Input DataFrame:")
    print(input_df)
    print("Data Types:")
    print(input_df.dtypes)
    
    # Check for missing columns and add them if necessary
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Use appropriate default values
    
    # Ensure the column order matches the training data
    input_df = input_df[X.columns]
    
    # Print the adjusted DataFrame
    print("Adjusted DataFrame for prediction:")
    print(input_df)
    
    # Use the pipeline to preprocess and predict
    prediction = pipeline.predict(input_df)
    
    return "Heart disease present" if prediction[0] == 1 else "Heart disease not present"

# Predict using user input
result = predict_heart_disease(user_input)
result
