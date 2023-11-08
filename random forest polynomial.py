import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('C:/Users/cvs59/Downloads/FYP/heart (1).csv')

# Assign feature variables to X and target variable to y
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
        'Oldpeak', 'ST_Slope']]
y = df['HeartDisease']

# Encode categorical variables
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
categorical_transformer = ColumnTransformer([('onehot', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_categorical = categorical_transformer.fit_transform(X)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_categorical)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create a polynomial feature transformer
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# Create an instance of the RandomForestClassifier model
model = RandomForestClassifier()

# Create a pipeline that includes feature scaling, polynomial features, and the RandomForest model
pipeline = Pipeline([
    ('polynomial_features', poly_features),
    ('random_forest', model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

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
