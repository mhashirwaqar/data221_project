def train_network(X_train, y_train):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer

    # Set seed for reproducibility
    tf.random.set_seed(1)

    # Build model
    model = Sequential()

    # Input layer (30 features: Time + V1–V28 + Amount)
    model.add(InputLayer(input_shape=(X_train.shape[1],)))

    # Hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

    # Output layer (binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    return model