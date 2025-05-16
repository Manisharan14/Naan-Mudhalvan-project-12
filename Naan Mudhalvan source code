import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load Dataset
def load_data():
    df = pd.read_csv("data/heart_disease.csv")  # Adjust path if needed
    print("✅ Data Loaded Successfully.")
    return df

# 2. Preprocess Dataset
def preprocess_data(df, target_column="Heart Disease"):
    df = df.copy()

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]

    # Handle missing values if any
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Split X and y
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 3. Visualize Correlation Matrix
def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

# 4. Train & Evaluate Models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n🧠 Model: {name}")
        print(f"✅ Accuracy: {acc:.4f}")
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

# 5. Main Pipeline
def main():
    print("🚀 AI Disease Prediction Pipeline Starting...\n")

    df = load_data()
    print("\n📄 Dataset Preview:")
    print(df.head())

    plot_correlation(df)

    X, y = preprocess_data(df)
    train_models(X, y)

    print("\n🎯 Pipeline Complete.")

if __name__ == "__main__":
    main()
