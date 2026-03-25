# TEAM COMMENT: Please add your code inside this function so it can be easily imported to main.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_decision_tree(X_train, y_train, depth_values=None, cross_validation=5):
    # If no depth values are provided, use a default list
    if depth_values is None:
        depth_values = [3, 4, 5, 7, 10]

    # Initialize variables to track the best model
    best_f1 = -1
    best_depth = depth_values[0]

    # Try different values of max_depth
    for depth in depth_values:
        # Create a Decision Tree model with the current depth
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)

        # Evaluate the model using cross-validation (F1-score)
        scores = cross_val_score(dt, X_train, y_train, cv=cross_validation, scoring="f1")
        mean_f1 = np.mean(scores)

        # Update best depth if the current model performs better
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_depth = depth

    # Train the final model using the best depth found
    best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model