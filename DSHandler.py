from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class DSHandler:
	def __init__(self, resize):
		self.resize = resize
		self.rescale = 1./255

	# Set the data generator for the training dataset
	# Also set modifiers for data augmentation (shear and zoom)
	def get_train_generator(self, dataset):
		# horizontal_flip = True
		# zoom_range=0.05
		# shear_range = 0.3,
		train_datagen = ImageDataGenerator(
			rescale = self.rescale, 
			horizontal_flip = True)
		train_generator = train_datagen.flow_from_directory(
			dataset,
			target_size = self.resize,
			batch_size = 32)

#		img = load_img('./train/anger/anger_8.jpg')  # this is a PIL image
#		x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#		x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
#		i = 0
#		for batch in train_datagen.flow(x, batch_size=1,
#				save_to_dir='preview', save_prefix='generated', save_format='jpeg'):
#			i += 1
#			if i > 20:
#				break  # otherwise the generator would loop indefinitely
		return train_generator

	def get_generator(self, dataset):
		datagen = ImageDataGenerator(rescale = self.rescale)
		generator = datagen.flow_from_directory(
			dataset,
			target_size = self.resize,
			batch_size = 32)
		return generator
