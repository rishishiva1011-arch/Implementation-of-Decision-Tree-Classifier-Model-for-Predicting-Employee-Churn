# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Prepare the Dataset
Define employee data with features like satisfaction level, last evaluation, number of projects, monthly hours, tenure, work accident, promotion history, department, and salary. Set left as the target variable (1=Churn, 0=Stayed) and convert to a DataFrame.

Step 2: Encode Categorical Variables
Apply get_dummies on Departments and salary columns with drop_first=True to convert categorical values into numeric binary columns and avoid multicollinearity.

Step 3: Split Features & Target
Drop the left column to form feature matrix X. Assign the left column as target vector y representing whether an employee churned or stayed.

Step 4: Train-Test Split & Train Model
Split data into 75% training and 25% testing using train_test_split. Initialize DecisionTreeClassifier with entropy criterion and max_depth=4, then fit it on the training data to learn churn patterns.

Step 5: Predict, Evaluate & Visualize
Predict churn on the test set and print Confusion Matrix, Accuracy Score, and Classification Report. Plot the trained decision tree using plot_tree with filled nodes and class labels (Stayed, Churn) to interpret the model's decisions visually.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rishikesh S
RegisterNumber:  212225240118
*/
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37, 0.41, 0.10, 0.92],
    'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52, 0.50, 0.77, 0.89],
    'number_project': [2, 5, 7, 5, 2, 2, 6, 5],
    'average_monthly_hours': [157, 262, 272, 223, 159, 153, 247, 224],
    'time_spend_company': [3, 6, 4, 5, 3, 3, 4, 5],
    'Work_accident': [0, 0, 0, 0, 0, 0, 0, 0],
    'promotion_last_5years': [0, 0, 0, 0, 0, 0, 0, 0],
    'Departments': ['sales','accounting','hr','technical','support','management','marketing','product'],
    'salary': ['low','medium','medium','high','low','low','medium','high'],
    'left': [1, 1, 1, 1, 1, 0, 1, 0]  # Target variable: 1=Churn, 0=Stayed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Encode categorical variables
# ------------------------------
df = pd.get_dummies(df, columns=['Departments','salary'], drop_first=True)

# ------------------------------
# Step 3: Split into features and target
# ------------------------------
X = df.drop('left', axis=1)
y = df['left']

# ------------------------------
# Step 4: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 5: Create Decision Tree Classifier
# ------------------------------
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = dt_model.predict(X_test)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Visualize the decision tree
# ------------------------------
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['Stayed','Churn'], filled=True)
plt.show()
```

## Output:

<img width="617" height="382" alt="image" src="https://github.com/user-attachments/assets/9212e158-b30b-4027-a7b3-804ed7f730a7" />

<br>

<img width="1570" height="790" alt="image" src="https://github.com/user-attachments/assets/65afcfae-dbae-4835-a7c8-75fa18415680" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
