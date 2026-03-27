import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from logistic_regression import train_logistic
from knn_model import train_knn
from decision_tree import train_decision_tree
from neural_network import train_network

def load_data():
    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        "creditcard.csv"
    )

    return dataset

def preprocess(dataset):

    # Separate features and target
    # Dropping 'Time' as it does not provide meaningful information for fraud detection
    X = dataset.drop(["Class", "Time"], axis=1)
    y = dataset["Class"]

    # Train-test split (80% train, 20% test) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,     # 20% of data used for testing, 80% for training
        stratify=y,        # preserves class distribution (important for imbalanced fraud dataset)
        random_state=42    # ensures reproducible results (same split every run)
    )

    # Standardize features so distance-based and gradient-based models perform well
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate(model, X_test, y_test, name):

    print("\n------------------------------")
    print(" Evaluating:", name)
    print("------------------------------")

    # Get predictions
    y_pred = model.predict(X_test)

    # If output is probability (Neural Network), convert to binary
    if len(y_pred.shape) > 1 or y_pred.dtype != int:
        y_pred = (y_pred > 0.2).astype(int).flatten()

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print("Accuracy: ", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:   ", round(recall, 4))
    print("F1-score: ", round(f1, 4))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():

    # Calling Load_data Function
    dataset = load_data()

    # Checking the Dataset
    print("\nDataset Preview")
    print(dataset.head())

    # Exploring the data
    print("\nDataset Description")
    print(dataset.describe())

    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess(dataset)

    # Training all Models and storing them in Dictionary
    models = {
        "Logistic Regression": train_logistic(X_train_scaled, y_train),
        "KNN": train_knn(X_train_scaled, y_train),
        "Decision Tree": train_decision_tree(X_train_scaled, y_train),
        "Neural Network": train_network(X_train_scaled, y_train)
    }

    # Evaluate all models
    for name, model in models.items():
        evaluate(model, X_test_scaled, y_test, name)

# Running the main python file
if __name__ == "__main__":
    main()