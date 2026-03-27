from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_knn(X_train_scaled, y_train, k_values=None, cross_validation=5):

    # If no k values are provided, use a default list
    # These values control how many neighbors are considered (smaller = more sensitive, larger = smoother)

    if k_values is None:
        k_values = [3, 5, 7, 11, 15, 21]  # selected range suitable for larger dataset

    # Initialize variables to track the best model
    best_f1 = -1                          # Start with the lowest possible F1-score
    best_k = k_values[0]                  # Default k value

    # Try different values of k (number of neighbors)
    for k in k_values:

        # Create a KNN model with the current k
        knn = KNeighborsClassifier(
            n_neighbors=k,               # number of nearest neighbors considered
            weights='distance',          # closer neighbors have more influence (helps with imbalance)
            n_jobs=-1                    # uses all CPU cores (faster for large dataset)
        )

        # Evaluate the model using cross-validation (F1-score)
        # cross_val_score splits data into multiple folds and evaluates performance
        scores = cross_val_score(
            knn,
            X_train_scaled,
            y_train,
            cv=cross_validation,         # number of folds
            scoring='f1'                 # F1-score is best for imbalanced classification
        )

        mean_f1 = np.mean(scores)        # Average F1-score across all folds

        # Update the best k if the current model performs better
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    # Train the final model using the best k found
    best_model = KNeighborsClassifier(
        n_neighbors=best_k,             # Use optimal k found from tuning
        weights='distance',             # must match training configuration
        n_jobs=-1                       # ensures faster computation
    )

    # Train model on full training dataset
    best_model.fit(X_train_scaled, y_train)

    return best_model