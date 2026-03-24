from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
def train_knn(X_train_scaled, y_train, k_values = None, cross_validation = 5):
    best_f1 = 0
    best_k = k_values[0]  # default to first value

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=cross_validation, scoring='f1')
        mean_f1 = np.mean(scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train_scaled, y_train)
    return best_model, best_k
