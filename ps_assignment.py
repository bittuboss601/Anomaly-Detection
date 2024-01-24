import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def explore_data(data):
    print(data.info())
    print(data.isnull().sum())
    print(data.describe())
    # Additional EDA if needed

def preprocess_data(data):
    # Convert object columns to numeric
    numeric_columns = data.select_dtypes(include=['float64']).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Separate categorical and numeric columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    numeric_columns = data.select_dtypes(include=['float64']).columns

    # Handle missing values for categorical columns with most frequent value
    for col in categorical_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Handle missing values for numeric columns with mean
    imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

    return data

def feature_engineering(data):
    # Drop irrelevant columns
    data = data.drop(['ID'], axis=1)

    # Convert all columns to strings for uniform encoding
    data = data.astype(str)

    # Convert categorical variables to dummy variables
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_columns = pd.DataFrame(encoder.fit_transform(data))
    data = pd.concat([data, encoded_columns], axis=1)
    data = data.drop(data.select_dtypes(include=['object']).columns, axis=1)

    return data

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def handle_imbalanced_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def train_model(X_train_resampled, y_train_resampled):
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_resampled, y_train_resampled)
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    # Load datasets
    data = load_data('./content/Dataset.csv')
    data_dict = load_data('./content/Data_Dictionary.csv')

    # Exploratory Data Analysis
    explore_data(data)

    # Data Preprocessing
    data = preprocess_data(data)

    # Feature Engineering
    data = feature_engineering(data)

    # Model Development
    X = data.drop('Default', axis=1)
    y = data['Default']
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Handle Imbalanced Data
    X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)

    # Train Model
    model = train_model(X_train_resampled, y_train_resampled)

    # Evaluate Model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
