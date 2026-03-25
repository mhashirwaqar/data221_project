# MAIN FILE TO LOAD DATASET & RUN THE MODELS

def load_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        "creditcard.csv"
    )

    return dataset

#Adding Preprocessing function - Mudit

def preprocess(dataset):
    """
    Preprocess the dataset:
    - Separate features (X) and target (y)
    - Perform train-test split (80:20) with stratification
    - Apply standard scaling
    - Return processed train and test sets
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Separate features and target
    X = dataset.drop("Class", axis=1)  # all columns except target
    y = dataset["Class"]               # target column

    # Train-test split (80% train, 20% test) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Apply standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

    #Added preprocessing function - Mudit


def evaluate(model, X_test, y_test, name):
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print("\n------------------------------")
    print(" Evaluating:", name)
    print("------------------------------")

    # Get predictions
    y_pred = model.predict(X_test)

    # If output is probabilities (Neural Network / Logistic), convert to 0/1
    y_pred = (y_pred > 0.5).astype(int).flatten()

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

    # Classification Report (nice table)
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
    X_train, X_test, y_train, y_test = preprocess(dataset)

    # Importing all the models from different files
    from logistic_regression import train_logistic
    from knn_model import train_knn
    from decision_tree import train_tree
    from neural_network import train_network

    # Training all Models and storing them in Dictionary
    models = {
        "Logistic Regression": train_logistic(X_train, y_train),
        "KNN": train_knn(X_train, y_train),
        "Decision Tree": train_tree(X_train, y_train),
        "Neural Network": train_network(X_train, y_train)
    }

    # Evaluate all models
    for name, model in models.items():
        evaluate(model, X_test, y_test, name)

# --------------

# Running the main python file
if __name__ == "__main__":
    main()
