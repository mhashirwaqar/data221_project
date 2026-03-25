# logistic_regression.py

# Import Logistic Regression model from sklearn
from sklearn.linear_model import LogisticRegression

def train_logistic(X_train_scaled, y_train):
    """
    Train a Logistic Regression model.

    Parameters:
    X_train_scaled : Scaled training feature data
    y_train : Training labels

    Returns:
    model : Trained Logistic Regression model
    """

    # Use class_weight='balanced' because dataset is highly imbalanced (fraud detection)
    model = LogisticRegression(class_weight='balanced')

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model
