def train_network(X_train, y_train):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer
    from sklearn.utils import class_weight
    import numpy as np

    # Set random seed for reproducibility (ensures same results every run)
    tf.random.set_seed(1)

    # Compute Class Weights (IMPORTANT for imbalanced data)
    # The dataset is highly imbalanced (very few fraud cases), so we assign higher weight to the minority class (fraud)
    # This helps the model pay more attention to fraud cases

    weights = class_weight.compute_class_weight(
        class_weight='balanced',        # Automatically balance weights
        classes=np.unique(y_train),     # Unique class labels (0 and 1)
        y=y_train                       # Target values
    )

    # Convert weights into dictionary format required by Keras
    class_weights = {
        0: weights[0],   # Weight for non-fraud class
        1: weights[1]    # Weight for fraud class (higher weight)
    }

    # Building Neural Network Model

    # Sequential model = layers stacked one after another
    model = Sequential()

    # Defining the input layer; shape of input data (number of features)
    model.add(InputLayer(input_shape=(X_train.shape[1],)))

    # Hidden Layer 1
    # 64 neurons with ReLU activation, (It helps learn non-linear patterns in the data)
    # More neurons gives the model more capacity to learn complex fraud patterns
    model.add(Dense(64, activation='relu'))

    # Hidden Layer 2 (32 neurons)
    # Gradually reducing size helps the model learn more refined representations
    model.add(Dense(32, activation='relu'))

    # Hidden Layer 3 (16 neurons)
    # Further reduction to simplify learned features before final prediction
    model.add(Dense(16, activation='relu'))

    # Output Layer
    # 1 neuron with sigmoid activation, (Outputs probability between 0 and 1 (fraud vs non-fraud))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the Model
    model.compile(
        optimizer='adam',                                   # efficient optimization algorithm
        loss='binary_crossentropy',                         # suitable for binary classification
        metrics=[                                           # track model performance during training
            'accuracy',                                     # Overall correctness
            tf.keras.metrics.Precision(name="precision"),   # How many predicted frauds are correct
            tf.keras.metrics.Recall(name="recall")          # How many actual frauds are detected
        ]
    )

    # Training Model

    model.fit(
        X_train,
        y_train,
        epochs=10,                           # number of passes through the dataset
        batch_size=32,                       # number of samples per training step
        verbose=1,                           # display training progress
        class_weight=class_weights           # applies higher importance to fraud cases
    )

    return model