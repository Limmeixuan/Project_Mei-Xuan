import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv('C:/Users/cvs59/Downloads/FYP/heart (1).csv')

# Assign feature variables to X and target variable to y
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
        'Oldpeak', 'ST_Slope']]
y = df['HeartDisease']

# Encode categorical variables
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
categorical_transformer = ColumnTransformer([('onehot', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = categorical_transformer.fit_transform(X)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create instances of models
rf_model = RandomForestClassifier()
logreg_model = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000 or any other suitable value
dt_model = DecisionTreeClassifier()
svm_model = SVC(probability=True)  # Enable probability estimation for SVM

# Create a Voting Classifier ensemble with soft voting
ensemble_model = VotingClassifier(estimators=[('RandomForest', rf_model),
                                              ('LogisticRegression', logreg_model),
                                              ('DecisionTree', dt_model),
                                              ('SVM', svm_model)],
                                  voting='soft')  # Use 'soft' voting for class probabilities

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions on the test set using the ensemble
y_pred_ensemble = ensemble_model.predict(X_test)

# Calculate evaluation metrics for the ensemble
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)

# Print the evaluation metrics for the ensemble
print("Ensemble Accuracy:", accuracy_ensemble)
print("Ensemble Precision:", precision_ensemble)
print("Ensemble Recall:", recall_ensemble)
print("Ensemble F1 Score:", f1_ensemble)
