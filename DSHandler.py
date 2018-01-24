from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class DSHandler:
    def __init__(self, resize):
        self.resize = resize
        self.rescale = 1. / 255

    # Set the data generator for the training dataset
    # Also set modifiers for data augmentation (shear and zoom)
    def get_train_generator(self, image_directory):
        # Creates generator for training set and applies data augmentation parameters
        train_datagen = ImageDataGenerator(
            rescale=self.rescale,
            horizontal_flip=True,
            zoom_range=0.05,
            shear_range=0.3)
        train_generator = train_datagen.flow_from_directory(
            image_directory,
            target_size=self.resize,
            batch_size=32)
        return train_generator

    # Set the data generator for validation and testing set
    # Also set modifiers for data augmentation (shear and zoom)
    def get_generator(self, image_directory):
        datagen = ImageDataGenerator(rescale=self.rescale)
        generator = datagen.flow_from_directory(
            image_directory,
            target_size=self.resize,
            batch_size=32)
        return generator
