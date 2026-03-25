def train_network(X_train, y_train):

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer

    # Set seed
    tf.random.set_seed(1)

    # Create model
    model = Sequential()

    return model