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

def preprocess(dataset):

    # separate features and target column
    # perform train test split with stratification
    # apply standard scaling
    # return training and testing data

    return 0, 0, 0, 0


def evaluate(model, X_test, y_test, name):

    # make predictions using trained model
    # compute confusion matrix
    # compute classification report
    # print results with model name

    return 0


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