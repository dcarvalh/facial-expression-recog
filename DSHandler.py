from keras.preprocessing.image import ImageDataGenerator

class DSHandler:
	def __init__(self, resize):
		self.resize = resize
		self.rescale = 1./255

	# Set the data generator for the training dataset
	# Also set modifiers for data augmentation (shear and zoom)
	def get_train_generator(self, dataset):
		train_datagen = ImageDataGenerator(
			rescale = self.rescale, 
			shear_range=0.2)
		train_generator = train_datagen.flow_from_directory(
			dataset,
			target_size = self.resize,
			batch_size = 32)
		return train_generator

	def get_generator(self, dataset):
		datagen = ImageDataGenerator(rescale = self.rescale)
		generator = datagen.flow_from_directory(
			dataset,
			target_size = self.resize,
			batch_size = 32)
		return generator
