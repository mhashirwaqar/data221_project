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

    # Checking the Shape of Dataset
    print("\nShape of Dataset")
    print(dataset.head())

    # Exploring the data
    print("\nDataset Features")
    print(dataset.describe())

    # step 2: preprocess data
    # call preprocess()

    # step 3: train models
    # call:
    # train_logistic()
    # train_knn()
    # train_tree()
    # train_network()

    # store all trained models (e.g., in dictionary)

    # step 4: evaluate each model
    # loop through models and call evaluate()

    return 0

if __name__ == "__main__":
    main()