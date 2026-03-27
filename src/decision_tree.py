from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_decision_tree(X_train, y_train, depth_values=None, cross_validation=5):

    # If no depth values are provided, use a default list
    # These values control tree complexity (smaller = simpler, larger = more complex)

    if depth_values is None:
        depth_values = [3, 5, 7, 10, 15]  # selected range to balance underfitting and overfitting

    # Initialize variables to track the best model
    best_f1 = -1                          # Start with the lowest possible F1-score
    best_depth = depth_values[0]          # Default depth

    # Try different values of max_depth
    for depth in depth_values:

        # Create a Decision Tree model with the current depth
        dt = DecisionTreeClassifier(
            max_depth=depth,         # controls how deep the tree can grow
            min_samples_leaf=5,      # Minimum samples required at a leaf node (prevents overfitting by avoiding very small splits)
            random_state=42,         # ensures reproducible results
            class_weight='balanced'  # handles imbalanced dataset (fraud detection)
        )

        # Evaluate the model using cross-validation (F1-score)
        # cross_val_score splits data into multiple folds and evaluates performance
        scores = cross_val_score(
            dt,
            X_train,
            y_train,
            cv=cross_validation,          # number of folds
            scoring="f1"                  # F1-score is best for imbalanced classification
        )

        mean_f1 = np.mean(scores)         # Average F1-score across all folds

        # Update the best depth if the current model performs better
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_depth = depth

    # Train the final model using the best depth found
    best_model = DecisionTreeClassifier(
        max_depth=best_depth,             # Use optimal depth found from tuning
        min_samples_leaf=5,               # Minimum samples required at a leaf node (prevents overfitting by avoiding very small splits)
        random_state=42,                  # ensures reproducible results
        class_weight='balanced'           # handles imbalanced dataset (fraud detection)
    )

    # Train model on full training dataset
    best_model.fit(X_train, y_train)

    return best_model