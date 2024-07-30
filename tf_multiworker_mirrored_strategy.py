from memory_profiler import profile


def train_func():
    """
    Trains a simple neural network (SimpleNN) for binary classification.

    The SimpleNN model is comprised of two dense layers with 10,000 and 1 units respectively.
    It uses the Adam optimizer with a learning rate of 0.01 to minimize binary cross-entropy loss.
    The training data consists of randomly generated input features (X) and corresponding binary target labels (y).

    Parameters: None

    Returns: None
    """
    import tensorflow as tf
    import numpy as np
    import json
    import os
    import tensorflow_datasets as tfds
    from loguru import logger
    # utils
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    def mnist_dataset(batch_size, buffer_size):
        datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

        mnist_train, mnist_test = datasets['train'], datasets['test']
        train_dataset = mnist_train.map(scale).cache().shuffle(buffer_size).batch(batch_size)
        eval_dataset = mnist_test.map(scale).batch(batch_size)
        return train_dataset, eval_dataset

    def build_and_compile_cnn_model(strategy):
        with strategy.scope():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)
            ])

            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=['accuracy'])
            return model

    tf_config_json = os.environ.get("TF_CONFIG")

    if tf_config_json is not None:
        # Distributed training
        import tensorflow as tf

        # Auto shard data for training
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # communication bwt clusters: [RING, NCCL and AUTO]
        communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.AUTO)

        # Create Strategy
        strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

        # Prepare datasets
        BUFFER_SIZE = 10000

        BATCH_SIZE_PER_REPLICA = 64
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        training_data, testing_data = mnist_dataset(BATCH_SIZE, BUFFER_SIZE)
        training_data = training_data.with_options(options)
        testing_data = testing_data.with_options(options)

        # Prepare model with appropriate strategy
        model = build_and_compile_cnn_model(strategy)

        # Training
        model.fit(training_data, epochs=3, steps_per_epoch=70)

    else:
        # Training at local
        batch_size = 64
        single_worker_dataset = mnist_dataset(batch_size)
        single_worker_model = build_and_compile_cnn_model()
        single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)


if __name__ == "__main__":
    import os
    from loguru import logger

    env = "prod"

    if env == "development":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.pop('TF_CONFIG', None)
        train_func()
    else:
        from kubeflow.training import TrainingClient, constants
        from datetime import datetime

        job_date = datetime.utcnow().strftime("%m-%d-%Y-%H-%M-%S")

        training_client = TrainingClient(
            context="mle",
            namespace="kubeflow",
            job_kind=constants.TFJOB_KIND
        )

        # result = training_client.delete_job(
        #     name="tensorflow-distributed-v2",
        # )
        job_name = f"tensorflow-distributed-{job_date}"
        result = training_client.create_job(
            name=job_name,
            train_func=train_func,
            base_image="tensorflow/tensorflow:2.15.0",
            num_workers=3,
            resources_per_worker={"cpu": "1"},
            packages_to_install=["numpy", "pandas", "loguru", "tensorflow_datasets"],
        )
        job_object = training_client.wait_for_job_conditions(name=job_name)
        if job_object is not None:
            logger.info(f"{job_name} completed")

