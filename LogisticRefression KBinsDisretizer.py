import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('C:/Users/cvs59/Downloads/FYP/heart (1).csv')

# Assign feature variables to X and target variable to y
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
        'Oldpeak', 'ST_Slope']]
y = df['HeartDisease']

# Define which features are continuous and categorical
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
continuous_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Encode categorical variables
categorical_transformer = ColumnTransformer([('onehot', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_categorical = categorical_transformer.fit_transform(X)

# Fill missing values with the mean for continuous features
imputer = SimpleImputer(strategy='mean')
X_continuous = imputer.fit_transform(X[continuous_features])

# Discretize the continuous features using KBinsDiscretizer
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
X_discretized = discretizer.fit_transform(X_continuous)

# Combine the categorical and discretized continuous features
X_final = np.hstack((X_categorical, X_discretized))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression model with different hyperparameters
model = LogisticRegression(solver='liblinear', C=0.01, class_weight='balanced')  # Adjust solver, C, and class_weight

# Train the model using the scaled training data
model.fit(X_train, y_train)

# Make predictions on the scaled test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
