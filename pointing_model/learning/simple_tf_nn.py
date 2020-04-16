import tensorflow as tf
from .model_interface import PointingMlModel
import datetime
import pointing_model.utils as utils


class SimpleTFNN(PointingMlModel):

    def model(self, X):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def train(self, X, y, **kwargs):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = self.model(X)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        log_dir = ".\\logs\\fit\\" +\
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        utils.ensure_dir_exists(log_dir, is_file=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        model.fit(
            x=x_train, y=y_train,
            epochs=5, validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback]
        )
