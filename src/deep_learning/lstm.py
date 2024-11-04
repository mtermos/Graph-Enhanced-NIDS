import os
import time
import numpy as np
import json
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import timeseries_dataset_from_array
from keras.regularizers import L2
from keras.optimizers import Adam

from model import Model


class MyLSTM(Model):
    def __init__(
            self,
            sequence_length,
            input_dim,
            dataset_name,
            stride=1,
            use_generator=False,
            already_sequenced=False,
            lstm_cells=[80],
            fc_cells=[80],
            num_classes=2,
            multi_class=False,
            network_features=[],
            loss="mse",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
            batch_size=128,
            epochs=25,
            early_stop_patience=10
    ):
        # Initialize the parent Model class
        super().__init__(sequential=True, multi_class=multi_class)

        # Model configuration
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.dataset_name = dataset_name
        self.stride = stride
        self.use_generator = use_generator
        self.already_sequenced = already_sequenced
        self.lstm_cells = lstm_cells
        self.fc_cells = fc_cells
        self.num_classes = num_classes
        self.network_features = network_features
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

    def model_name(self):
        # Generate a name for the model based on its configuration
        classification = "mc" if self.multi_class else "bc"
        network_features_string = "-".join(
            ["nf"] + self.network_features) if self.network_features else ""
        layers_name = "-".join(map(str, self.cells))
        return f"lstm sl-{self.sequence_length} {classification} layers-{network_features_string}-{layers_name}"

    def build(self):
        LAMBD = 0.01  # L2 regularization parameter

        self.model = Sequential()

        for i, c in enumerate(self.lstm_cells):
            return_seq = True if i != len(self.lstm_cells) - 1 else False
            if i == 0:
                self.model.add(layers.LSTM(
                    units=c,
                    activation='relu',
                    input_shape=(self.sequence_length, self.input_dim),
                    kernel_regularizer=L2(LAMBD),
                    recurrent_regularizer=L2(LAMBD),
                    bias_regularizer=L2(LAMBD),
                    return_sequences=return_seq
                ))
            else:
                self.model.add(layers.LSTM(
                    units=c,
                    activation='relu',
                    kernel_regularizer=L2(LAMBD),
                    recurrent_regularizer=L2(LAMBD),
                    bias_regularizer=L2(LAMBD),
                    return_sequences=return_seq
                ))

            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(0.2))

        for i, c in enumerate(self.fc_cells):
            self.model.add(layers.Dense(
                units=c,
                kernel_regularizer=L2(LAMBD),
                bias_regularizer=L2(LAMBD),
                activation='relu'
            ))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.Dropout(0.2))

        if self.multi_class:
            self.model.add(layers.Dense(
                self.num_classes, activation='softmax'))
            self.model.compile(
                optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        else:
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.compile(optimizer=self.optimizer,
                               loss='binary_crossentropy', metrics=['accuracy'])

    def create_generator(self, data, labels, batch_size):
        # Create a data generator for training
        return timeseries_dataset_from_array(
            data=data,
            targets=labels[self.sequence_length - 1:],
            sequence_length=self.sequence_length,
            sequence_stride=self.stride,
            shuffle=False,
            batch_size=batch_size
        )

    def create_sequences(self, data, labels):

        if self.sequence_length == 1:
            data_reshaped = np.reshape(
                data, (data.shape[0], 1, data.shape[1]))
            return np.array(data_reshaped), np.array(labels)

        # Create sequences for training
        data_reshaped = [data[i - self.sequence_length:i]
                         for i in range(self.sequence_length, len(data) + 1)]
        labels_reshaped = labels[self.sequence_length - 1:]
        return np.array(data_reshaped), np.array(labels_reshaped)

    def train(self, training_data, training_labels):

        id_time = time.strftime("%Y%m%d-%H%M%S")
        weights_folder = f"models/weights/{self.dataset_name}/lstm/{id_time}"
        logs_folder = f"logs/scalars/{self.dataset_name}/lstm/{id_time}"

        self.makedirs_and_data(weights_folder, logs_folder)

        # Train the model
        if not self.model:
            self.build()

        model_path = f"models/weights/{self.dataset_name}/{self.model_name()}/best.hdf5"
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
        else:
            filepath = f"models/weights/{self.dataset_name}/{self.model_name()}/weights-improvement-{{epoch:02d}}-{{loss:.4f}}.hdf5"
            checkpoint = ModelCheckpoint(
                filepath, verbose=1, save_best_only=False, mode='max')
            early_stopping = EarlyStopping(
                monitor="loss", patience=self.early_stop_patience)

            tensorboard_callback = TensorBoard(log_dir=logs_folder)

            callbacks_list = [checkpoint, early_stopping, tensorboard_callback]

            if self.already_sequenced:
                self.model.fit(training_data, training_labels, epochs=self.epochs,
                               batch_size=self.batch_size, shuffle=False, callbacks=callbacks_list)
            elif self.use_generator:
                training_generator = self.create_generator(
                    training_data, training_labels, self.batch_size)
                self.model.fit(training_generator, epochs=self.epochs,
                               shuffle=False, callbacks=callbacks_list)
            else:
                x_train, y_train = self.create_sequences(
                    training_data, training_labels)
                self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                               shuffle=False, callbacks=callbacks_list)

    def predict(self, testing_data):
        # Predict using the model
        start = time.time()
        testing_labels = np.zeros(testing_data.shape[0])

        if self.already_sequenced:
            y_predictions = self.model.predict(testing_data)
        elif self.use_generator:
            testing_generator = self.create_generator(
                testing_data, testing_labels, self.batch_size)
            y_predictions = self.model.predict(testing_generator)
        else:
            x_test, _ = self.create_sequences(testing_data, testing_labels)
            y_predictions = self.model.predict(
                x_test, batch_size=self.batch_size)

        end = time.time()
        y_predictions = np.argmax(y_predictions, axis=1) if self.num_classes > 2 else (
            y_predictions >= 0.5).astype(int)
        return y_predictions, end - start

    def evaluate(self, predictions, labels, time, verbose=0):
        labels = np.array(labels[self.sequence_length - 1:]
                          if not self.already_sequenced else labels)
        return super().evaluate(predictions, labels, time, verbose)


if __name__ == '__main__':
    myLSTM = MyLSTM(
        sequence_length=10,
        input_dim=5,
        dataset_name="test_dataset",
        lstm_cells=[50, 20],
        num_classes=2,
        multi_class=False,
        batch_size=32,
        epochs=10
    )

    id_time = time.strftime("%Y%m%d-%H%M%S")
    weights_folder = f"weights/{id_time}"
    logs_folder = f"logs/{id_time}"
    myLSTM.build()
    myLSTM.makedirs_and_data(weights_folder, logs_folder)
