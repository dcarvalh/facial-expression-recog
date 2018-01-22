from keras.preprocessing.image import ImageDataGenerator

class DSHandler:
	def __init__(self):
		#### Set data generator for our datasets
		# Set the data generator for our training dataset
		# Also set modifiers for data augmentation (shear and zoom)
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2)
		self.train_generator = train_datagen.flow_from_directory('./train',
			target_size = (224, 224),
			batch_size = 32)

		# Set the data generator for the validation dataset
		validation_datagen = ImageDataGenerator(rescale = 1./255)
		self.validation_generator = validation_datagen.flow_from_directory(
			'./validation',
			target_size = (224, 224),
			batch_size = 32)

		# Set the data generator for the testing dataset
		test_datagen = ImageDataGenerator(rescale = 1./255)
		self.test_generator = test_datagen.flow_from_directory('./test',
			target_size = (224, 224),
			batch_size = 32)
