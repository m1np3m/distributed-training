def train_func():
    """
    Trains a simple neural network (SimpleNN) for binary classification.

    The SimpleNN model is comprised of two dense layers with 10,000 and 1 units respectively.
    It uses the Adam optimizer with a learning rate of 0.01 to minimize binary cross-entropy loss.
    The training data consists of randomly generated input features (X) and corresponding binary target labels (y).

    Parameters: None

    Returns: None
    """
    import numpy as np
    from tensorflow.keras.layers import Dense
    import keras
    from keras import Sequential
    from tensorflow.keras.optimizers import Adam

    def SimpleNN():
        model = Sequential()
        model.add(Dense(10000, input_dim=10, activation='relu'))  # 10 input features to 50 hidden units
        model.add(Dense(1, activation='sigmoid'))

        return model

    model = SimpleNN()
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    np.random.seed(0)

    # Generate random data
    X = np.random.rand(1000, 10)
    y = (np.sum(X, axis=1) > 5).astype(int)  # Binary target

    model.fit(X, y, epochs=10000, batch_size=64, verbose=1)


if __name__ == "__main__":
    from kubeflow.training import TrainingClient, constants
    from datetime import datetime

    job_name = datetime.utcnow().strftime("%m-%d-%Y-%H-%M-%S")

    training_client = TrainingClient(
        context="mle",
        namespace="kubeflow",
        job_kind=constants.PYTORCHJOB_KIND
    )

    result = training_client.create_job(
        name=f"tensorflow-distributed-{job_name}",
        train_func=train_func,
        base_image="tensorflow/tensorflow:2.17.0",
        num_workers=3,
        resources_per_worker={"cpu": "1"},
    )
