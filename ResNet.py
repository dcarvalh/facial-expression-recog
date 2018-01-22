from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

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


	##### First training
	## Do a first training of the model on the new data for a few epoch
	def first_train(self, train_gen, validation_gen):
		# first: train only the top layers, which were randomly initialized
		# i.e. freeze all convolutional ResNet layers
		for layer in self.base_model.layers:
		    layer.trainable = False

		# compile the model (do it *after* setting layers to non-trainable)
		self.model.compile(optimizer='rmsprop', 
				loss='categorical_crossentropy', 
				metrics=['accuracy'])

			
		# TODO  samples_per_e = size(training set)/batch_size
		self.model.fit_generator(train_gen, steps_per_epoch=1, epochs=1,
			validation_data = validation_gen) 


	##### Fine tuning
	# Here, the top layers are trained and we can start fine-tuning
	# convolutional layers from Resnet. Freeze the bottom N layers
	# and train the remaining top layers.
	def fine_tune(self, train_gen, validation_gen):
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
		self.model.fit_generator(train_gen, 
			steps_per_epoch=1, epochs=1,
			validation_data = validation_gen)

	# Evaluates our classifier on the testing set and print accuracy and loss
	def evaluate(self, test_gen):
		print(self.model.metrics_names)
		# TODO also call predict_generator to get the list of results
		print(self.model.evaluate_generator(test_gen))
