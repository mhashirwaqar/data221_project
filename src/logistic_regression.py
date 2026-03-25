# TODO: Implement Logistic Regression model here

from sklearn.linear_model import LogisticRegression

def train_logistic(X_train_scaled, y_train):
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    return model
