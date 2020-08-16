import tensorflow as tf
from tensorboard import program
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleNet, ComplexNet, SimpleCnn, ComplexCnn, CustomMobileNetV2, CustomVGG16


# Create datasets
training_size = 0.8
validation_size = 0.2
image_height = 96
image_width = 96
image_channels = 1
image_size = (image_height, image_width, image_channels)
directory = '../data/anuka1200'
logs = './logs/'

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=directory,
    subset='training',
    labels='inferred',
    validation_split=validation_size,
    seed=123,
    color_mode='grayscale',
    image_size=(image_height, image_width))

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=directory,
    subset='validation',
    labels='inferred',
    validation_split=validation_size,
    seed=123,
    color_mode='grayscale',
    image_size=(image_height, image_width))

class_names = train_dataset.class_names
print('Class Names', class_names)

# Explore data
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        image = np.array(images[i].numpy().astype("uint8"))
        if image_channels == 1:
            plt.imshow(image.squeeze(), cmap='gray')
        else:
            plt.imshow(image)
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


# Create model
def create_models():
    model_0 = SimpleNet(image_size)
    model_1 = ComplexNet(image_size)
    model_2 = SimpleCnn(image_size)
    model_3 = ComplexCnn(image_size)
    model_4 = CustomMobileNetV2(image_size)
    model_5 = CustomVGG16(image_size)

    return [
        model_0,
        model_1,
        model_2,
        model_3,
        model_4,
        model_5
    ]


# Train models
def train_models(models_list):
    for m in models_list:
        m.compile()
        m.train(train_dataset, validation_dataset)
        m.load_weights()
        m.evaluate(validation_dataset)


# Execute
models = create_models()
train_models(models)

print('------------ Results -----------------')
for model in models:
  print(f'{model.model_name}: validation_loss: {model.score[0]}, validation_accuracy:{model.score[1]}')

# Start Tensorboard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logs])
url = tb.launch()
