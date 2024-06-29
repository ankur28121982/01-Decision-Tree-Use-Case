import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# Load the dataset
file_path = os.path.join(os.getcwd(), 'data1.csv')
data = pd.read_csv(file_path)

# Separate features (X) and target variable (y)
X = data.drop(columns=['Employee ID', 'Efficiency (%)'])
y = data['Efficiency (%)']

# Perform one-hot encoding for categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load the new dataset
file_path_new = os.path.join(os.getcwd(), 'data2.csv')
data_new = pd.read_csv(file_path_new)

# Drop the Efficiency (%) column if present
if 'Efficiency (%)' in data_new.columns:
    data_new = data_new.drop(columns=['Efficiency (%)'])

# Convert categorical variables to numerical format using one-hot encoding
data_new_encoded = pd.get_dummies(data_new, columns=['Education Level'])

# Ensure the new dataset has the same columns as the training set
missing_cols = set(X_encoded.columns) - set(data_new_encoded.columns)
for col in missing_cols:
    data_new_encoded[col] = 0
data_new_encoded = data_new_encoded[X_encoded.columns]

# Use the trained decision tree model to make predictions on the new data
y_pred_new = clf.predict(data_new_encoded)

# Print the predictions
print("Predictions for data2.csv:")
print(y_pred_new)

# Add the predicted efficiency to the new data DataFrame
data_new['Predicted Efficiency (%)'] = y_pred_new

# Save the new DataFrame with predictions to a new CSV file
file_path_output = os.path.join(os.getcwd(), 'data2_with_predictions.csv')
data_new.to_csv(file_path_output, index=False)

print("Predictions saved to:", file_path_output)
