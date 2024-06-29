## Project Title

Decision Tree Classifier for Predicting Employee Efficiency

## Project Description

This project uses a Decision Tree Classifier to predict employee efficiency based on various features such as age, experience, education level, and average handling time. The model is trained on a dataset (`data1.csv`) and can make predictions on new datasets (e.g., `data2.csv`). 

## Dataset

### data1.csv

The initial dataset used for training the model contains the following columns:
- `Employee ID`: Unique identifier for each employee
- `Age`: Age of the employee
- `Experience (years)`: Number of years of experience the employee has
- `Education Level`: Educational qualification of the employee (Bachelor, Master's, etc.)
- `Average Handling Time (AHT)`: Average time taken by the employee to handle a task
- `Efficiency (%)`: Efficiency of the employee as a percentage

Sample data:
```
Employee ID, Age, Experience (years), Education Level, Average Handling Time (AHT), Efficiency (%)
1, 25, 3, Bachelor, 20, 85
2, 28, 5, Master's, 18, 90
```

### data2.csv

The new dataset used for prediction contains similar columns but may not include the `Efficiency (%)` column. 

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ankur28121982/Decision-Tree-Use-Case.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Ensure that the required packages are installed:
   ```bash
   pip install pandas scikit-learn
   ```
4. Prepare your datasets (`data1.csv` and `data2.csv`) and place them in the specified directory.

5. Run the script to train the model and make predictions:
   ```bash
   python script.py
   ```

## Script Overview

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = r"C:\Users\Ankur\Desktop\Data Science\Data Science\data1.csv"
data = pd.read_csv(file_path)

# Separate features (X) and target variable (y)
X = data.drop(columns=['Employee ID', 'Efficiency (%)'], inplace=False)
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
```

### Making Predictions

```python
# Load the new dataset
file_path_new = r"C:\Users\Ankur\Desktop\Data Science\Data Science\data2.csv"
data_new = pd.read_csv(file_path_new)

# Convert categorical variables to numerical format using one-hot encoding
data_new_encoded = pd.get_dummies(data_new, columns=['Education Level'])

# Drop unnecessary columns
X_new = data_new_encoded.drop(columns=['Employee ID', 'Efficiency (%)'], inplace=False)

# Use the trained decision tree model to make predictions on the new data
y_pred_new = clf.predict(X_new)

# Print the predictions
print("Predictions for data2.csv:")
print(y_pred_new)

# Add the predicted efficiency to the new data DataFrame
data_new['Predicted Efficiency (%)'] = y_pred_new

# Save the new DataFrame with predictions to a new CSV file
file_path_output = r"C:\Users\Ankur\Desktop\Data Science\Data Science\data2_with_predictions.csv"
data_new.to_csv(file_path_output, index=False)

print("Predictions saved to:", file_path_output)
```

## Author

- Dr. Ankur
- ankur1122@gmail.com

