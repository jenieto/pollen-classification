import tensorflow as tf
import numpy as np
from tensorboard import program
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from models import SimpleNet, ComplexNet, SimpleCnn, ComplexCnn, CustomMobileNetV2, CustomVGG16
import plots

# Create datasets
training_size = 0.8
validation_size = 0.2
image_height = 96
image_width = 96
image_channels = 1
image_size = (image_height, image_width, image_channels)
directory = '../data/anuka1200'
logs = './logs/'
data_augmentation = False
color_mode = 'grayscale'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

if data_augmentation:
    train_datagen = ImageDataGenerator(
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_size)
    test_datagen = ImageDataGenerator(validation_split=validation_size)

    train_dataset = train_datagen.flow_from_directory(
        directory=directory,
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='binary',
        subset='training',
        shuffle=True,
        color_mode=color_mode)
    validation_dataset = test_datagen.flow_from_directory(
        directory=directory,
        target_size=(image_height, image_width),
        batch_size=32,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        color_mode=color_mode)
    class_names = list(train_dataset.class_indices.keys())
    print('Class Names', class_names)
    images = train_dataset[0][0][0:9]
    plots.plot_images(images, image_channels)
else:
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        subset='training',
        labels='inferred',
        validation_split=validation_size,
        seed=123,
        color_mode='grayscale',
        shuffle=True,
        image_size=(image_height, image_width))
    # train_dataset = train_dataset.map(lambda x, label: (x / 255.0, label))

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        subset='validation',
        labels='inferred',
        validation_split=validation_size,
        seed=123,
        color_mode='grayscale',
        shuffle=True,
        image_size=(image_height, image_width))
    # validation_dataset = validation_dataset.map(lambda x, label: (x / 255.0, label))

    class_names = train_dataset.class_names
    print('Class Names', class_names)

    # Explore data
    plots.plot_dataset(train_dataset, class_names, image_channels)


# Create model
def create_models():
    model_1 = SimpleNet()
    model_2 = ComplexNet()
    model_3 = SimpleCnn()
    model_4 = ComplexCnn()
    model_5 = CustomMobileNetV2(image_size)
    model_6 = CustomVGG16(image_size)

    return [
        # model_1,
        # model_2,
        # model_3,
        # model_4,
        model_5,
        model_6
    ]


# Train models
def train_models(models_list):
    for m in models_list:
        m.compile()
        m.train(train_dataset, validation_dataset)
        m.load_weights()
        m.evaluate(validation_dataset)
        m.save_history()


# Execute
models = create_models()
train_models(models)

print('------------ Results -----------------')
for model in models:
    print(f'{model.model_name}: validation_loss: {model.score[0]}, validation_accuracy:{model.score[1]}')


'''for images, labels in validation_dataset.take(1):
    for i in range(20):
        predicted_result = models[0].model.predict(np.expand_dims(images[i], axis=0))
        # print(predicted_result)
        print(labels[i])

# Start Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logs])
url = tb.launch()
'''