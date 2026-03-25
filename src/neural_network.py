def train_network(X_train, y_train):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer

    # Set seed for reproducibility (same results every run)
    tf.random.set_seed(1)

    # Build a simple feedforward neural network
    model = Sequential()

    # Input layer
    # X_train.shape[1] = number of features (Time + V1–V28 + Amount)
    model.add(InputLayer(input_shape=(X_train.shape[1],)))

    # Hidden layer 1
    # 32 neurons with ReLU activation (standard choice for learning non-linear patterns)
    model.add(Dense(32, activation='relu'))

    # Hidden layer 2
    # 16 neurons (smaller layer to gradually reduce complexity)
    model.add(Dense(16, activation='relu'))

    # Output layer
    # 1 neuron with sigmoid: outputs probability (0 to 1) for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    # Adam optimizer: efficient and widely used
    # Binary crossentropy: suitable for binary classification (fraud vs non-fraud)
    # Accuracy: basic metric (but we will also use precision, recall, F1 later)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    # epochs=10: enough for learning without overfitting (dataset is large)
    # batch_size=32: standard choice for stable and efficient training
    # verbose=1: shows training progress
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Return trained model
    return model