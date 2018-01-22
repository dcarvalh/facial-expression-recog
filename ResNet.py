from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np

class ResNet:
	def __init__(self, number_classes):
		# create the base pre-trained model
		self.base_model = ResNet50(weights='imagenet', include_top=False)

		# add a global spatial average pooling layer
		layer = self.base_model.output
		layer = GlobalAveragePooling2D()(layer)
		# Add a fully-connected layer
		layer = Dense(1024, activation='relu')(layer)
		# and a logistic layer for our classes
		prediction = Dense(number_classes, activation='softmax')(layer)

		# this is the model we will train
		self.model = Model(inputs=self.base_model.input, outputs=prediction)

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

	##### First training
	## Do a first training of the model on the new data for a few epoch
	def first_train(self):
		# first: train only the top layers, which were randomly initialized
		# i.e. freeze all convolutional ResNet layers
		for layer in self.base_model.layers:
		    layer.trainable = False

		# compile the model (do it *after* setting layers to non-trainable)
		self.model.compile(optimizer='rmsprop', 
				loss='categorical_crossentropy', 
				metrics=['accuracy'])

			
		# TODO  samples_per_e = size(training set)/batch_size
		self.model.fit_generator(self.train_generator, steps_per_epoch=1, epochs=1,
			validation_data = self.validation_generator) 


	##### Fine tuning
	# Here, the top layers are trained and we can start fine-tuning
	# convolutional layers from Resnet. Freeze the bottom N layers
	# and train the remaining top layers.
	def fine_tune(self):
		# we chose to train the top 2 resnet blocks, i.e. we will freeze
		# the first 249 layers and unfreeze the rest:
		for layer in self.model.layers[:170]:
			layer.trainable = False

		for layer in self.model.layers[170:]:
			layer.trainable = True

		# Need to recompile the model for these changes to take effect
		# we use SGD with a low learning rate
		self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		# Train model again (this time fine-tuning the top 2 resnet blocks)
		# alongside the top Dense layers
		self.model.fit_generator(self.train_generator, 
			steps_per_epoch=1, epochs=1,
			validation_data = self.validation_generator)

	# Evaluates our classifier on the testing set and print accuracy and loss
	def evaluate(self):
		print(self.model.metrics_names)
		# TODO also call predict_generator to get the list of results
		print(self.model.evaluate_generator(self.test_generator))
