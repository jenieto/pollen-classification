import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np

history_path = './logs/history'


def plot_result_histories(model_names=[], histories=[]):
    legend = []
    for (name, history) in zip(model_names, histories):
        legend.append(name)
        plt.plot(history['val_binary_accuracy'])
        plt.title('Comparacion de modelos')
        plt.ylabel('validation accuracy')
        plt.xlabel('epoch')
    plt.legend(legend, loc='lower right')
    plt.show()
    plt.clf()

    for (name, history) in zip(model_names, histories):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title(f'Loss {name}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'val_loss'], loc='lower right')
        plt.ylim(0, 1)
        plt.show()
        plt.clf()

        plt.plot(history['binary_accuracy'])
        plt.plot(history['val_binary_accuracy'])
        plt.title(f'Accuracy {name}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy', 'val_accuracy'], loc='lower right')
        plt.show()
        plt.clf()


def plot_dataset(dataset, class_names=[], image_channels=1):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
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


def plot_images(images=[], image_channels=1):
    for (i, image) in enumerate(images):
        ax = plt.subplot(3, 3, i+1)
        if image_channels == 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.axis("off")
    plt.show()


def plot_results_from_logs():
    path = f'{history_path}/*.pickle'
    model_names = []
    histories = []
    for filename in glob.glob(path):
        name = filename[len(history_path)+1:filename.index('-')]
        history = pickle.load(open(filename, 'rb'))
        model_names.append(name)
        histories.append(history)
    plot_result_histories(model_names, histories)


if __name__ == '__main__':
    plot_results_from_logs()