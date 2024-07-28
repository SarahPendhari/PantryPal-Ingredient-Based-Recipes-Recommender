import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
file_path = 'dataset\Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
data = pd.read_csv(file_path)

# Display column names to confirm their exact names
print("Column Names:")
print(data.columns)

# Check the first few rows to understand the structure
print("\nDataset Preview:")
print(data.head())

# Data Cleaning

# Check for missing values
print("\nMissing values in Dataset:")
print(data.isnull().sum())

# Handle missing values (dropping rows with missing values)
data.dropna(subset=['Ingredients', 'Cleaned_Ingredients'], inplace=True)

# Convert ingredient lists to a single string
data['Cleaned_Ingredients'] = data['Cleaned_Ingredients'].apply(lambda x: ' '.join(eval(x)))

# Split data into training and testing sets
X = data['Cleaned_Ingredients']
y = data['Title']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
os.makedirs('preprocessed_data', exist_ok=True)
train_data = pd.DataFrame({'ingredients': X_train, 'recipe_name': y_train})
test_data = pd.DataFrame({'ingredients': X_test, 'recipe_name': y_test})

train_data.to_csv('preprocessed_data/train_data.csv', index=False)
test_data.to_csv('preprocessed_data/test_data.csv', index=False)

print("Data preprocessing complete and saved to preprocessed_data/")
