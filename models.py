import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2, VGG16
from keras.utils.vis_utils import plot_model
from datetime import datetime
from IPython.display import Image


class NeuralNet:
    epochs = 30
    history = None
    score = None
    model = None
    models_directory = 'models'
    model_name = ''
    logs = './logs/'

    def __callbacks__(self):
        filepath_mdl = f'{self.models_directory}/{self.model_name}.h5'
        checkpoint = ModelCheckpoint(filepath_mdl, monitor='val_binary_accuracy', verbose=1, save_best_only=True)
        log_dir = f"{self.logs}/fit/{self.model_name}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensor_board = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience=10, verbose=1)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=5, verbose=1, min_lr=0.0001)
        return [checkpoint, tensor_board, reduce_lr_loss]

    @staticmethod
    def _reshape_dataset(dataset):
        return dataset.map(lambda t, y: (tf.repeat(input=t, repeats=3, axis=-1), y))

    def compile(self):
        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    def train(self, train_dataset, validation_dataset):
        print('-----------------------------------------------------------------------')
        print(f'Fitting model {self.model_name}')
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.epochs,
            callbacks=self.__callbacks__())

    def display_model(self):
        plot_file = f'models/{self.model_name}.png'
        plot_model(self.model, to_file=plot_file, show_shapes=True, show_layer_names=True)
        Image(retina=True, filename=plot_file)

    def load_weights(self):
        self.model.load_weights(f'{self.models_directory}/{self.model_name}.h5')

    def evaluate(self, validation_dataset):
        self.score = self.model.evaluate(validation_dataset)

    def save_history(self):
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f'{self.logs}/history/{self.model_name}-{date}.pickle', 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)


class SimpleNet(NeuralNet):
    model_name = 'SimpleNet'

    def __init__(self,):
        self.model = Sequential([
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
          ])


class ComplexNet(NeuralNet):
    model_name = 'ComplexNet'

    def __init__(self):
        self.model = Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])


class SimpleCnn(NeuralNet):
    model_name = 'SimpleCNN'

    def __init__(self):
        self.model = Sequential([
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.50),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])


class ComplexCnn(NeuralNet):
    model_name = 'ComplexCnn'

    def __init__(self):
        self.model = Sequential([
            layers.SeparableConv2D(64, 3, padding='same', activation='relu'),
            layers.AveragePooling2D(),
            layers.Dropout(0.50),
            layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
            layers.AveragePooling2D(),
            layers.Dropout(0.50),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
          ])


class CustomMobileNetV2(NeuralNet):
    model_name = 'MobileNetV2'

    def __init__(self, image_size):
        adapted_image_size = (image_size[0], image_size[1], 3)
        mobile_net = MobileNetV2(weights='imagenet', include_top=False, input_shape=adapted_image_size, pooling='avg')
        for layer in mobile_net.layers:
            layer.trainable = False
        self.model = tf.keras.models.Sequential([
            mobile_net,
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])


class CustomVGG16(NeuralNet):
    model_name = 'VGG16'

    def __init__(self, image_size):
        adapted_image_size = (image_size[0], image_size[1], 3)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=adapted_image_size, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        self.model = tf.keras.models.Sequential([
            base_model,
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
