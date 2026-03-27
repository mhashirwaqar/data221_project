from sklearn.linear_model import LogisticRegression

def train_logistic(X_train_scaled, y_train):

    # Create Logistic Regression model
    model = LogisticRegression(
        class_weight='balanced',  # handles imbalanced dataset (fraud detection)
        max_iter=1000,            # ensures convergence on large dataset
        penalty='l2'              # standard regularization to prevent overfitting
    )

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model