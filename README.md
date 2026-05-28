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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Employee.csv")

label = LabelEncoder()

df['Work_accident'] = label.fit_transform(df['Work_accident'])
df['promotion_last_5years'] = label.fit_transform(df['promotion_last_5years'])
df['Departments '] = label.fit_transform(df['Departments '])
df['salary'] = label.fit_transform(df['salary'])
df['left'] = label.fit_transform(df['left'])

X = df[['satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company']]

y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=X.columns,
          class_names=['Stay', 'Left'],
          filled=True)

plt.title("Employee Churn Decision Tree")

plt.show()
```

## Output:

<img width="760" height="213" alt="Screenshot 2026-05-28 150619" src="https://github.com/user-attachments/assets/4f1ffbb8-ebb7-4fe0-b776-fd1aa2fea061" />


<br>

<img width="960" height="658" alt="image" src="https://github.com/user-attachments/assets/91205e65-b3ab-4183-9b53-183cac3b9bbc" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
