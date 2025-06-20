# supervised_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_data():
    """Loads the Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names

def preprocess_data(df):
    """Handles missing values (if any) and normalizes the data."""
    # No missing values in Iris dataset, but including for completeness
    # For real-world data, you might use df.fillna() or other strategies

    # Separate features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def train_and_evaluate(X_train, X_test, y_train, y_test, target_names):
    """Trains and evaluates Decision Tree and Random Forest classifiers."""
    
    # Decision Tree Classifier
    print("\n--- Decision Tree Classifier ---")
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    y_pred_dt = dt_classifier.predict(X_test)
    
    print("Classification Report (Decision Tree):")
    print(classification_report(y_test, y_pred_dt, target_names=target_names))
    
    print("Cross-validation scores (Decision Tree):")
    cv_scores_dt = cross_val_score(dt_classifier, X_train, y_train, cv=5)
    print(f"Mean accuracy: {cv_scores_dt.mean():.2f} (+/- {cv_scores_dt.std() * 2:.2f})")
    
    plot_confusion_matrix(y_test, y_pred_dt, target_names, "Decision Tree Confusion Matrix")

    # Random Forest Classifier
    print("\n--- Random Forest Classifier ---")
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred_rf = rf_classifier.predict(X_test)
    
    print("Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    
    print("Cross-validation scores (Random Forest):")
    cv_scores_rf = cross_val_score(rf_classifier, X_train, y_train, cv=5)
    print(f"Mean accuracy: {cv_scores_rf.mean():.2f} (+/- {cv_scores_rf.std() * 2:.2f})")
    
    plot_confusion_matrix(y_test, y_pred_rf, target_names, "Random Forest Confusion Matrix")

def plot_confusion_matrix(y_true, y_pred, target_names, title):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saved {title.replace(' ', '_')}.png")

if __name__ == "__main__":
    # 1. Load the dataset
    print("Loading dataset...")
    data_df, target_names = load_data()
    print("Dataset loaded successfully.")
    print("Dataset head:\n", data_df.head())

    # 2. Preprocess the data
    print("Preprocessing data...")
    X, y = preprocess_data(data_df)
    print("Data preprocessing complete.")
    print("Scaled features head:\n", X.head())

    # 3. Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # 4. Train and evaluate classifiers
    train_and_evaluate(X_train, X_test, y_train, y_test, target_names)


