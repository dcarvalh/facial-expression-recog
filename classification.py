from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np

from handle_data import generate_tuples


number_classes = 7

############ Finetuning

##### Model definition
# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer for our classes
predictions = Dense(number_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


##### First training

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (do it *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
		metrics=['accuracy'])



####  Do a first training of the model on the new data for a few epochs

# Set the data generator for our training dataset
# Also set modifiers for data augmentation (shear and zoom)
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2)
train_generator = train_datagen.flow_from_directory('./train',
	target_size = (224, 224),
	batch_size = 32)

# Set the data generator for the validation dataset
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory('./validation',
	target_size = (224, 224),
	batch_size = 32)
	
# TODO  samples_per_e = size(training set)/batch_size
print("First training...")
model.fit_generator(train_generator, steps_per_epoch=1, epochs=1,
	validation_data = validation_generator) 

##### Fine tuning

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from Resnet. We will freeze the bottom N layers
# and train the remaining top layers.

# we chose to train the top 2 resnet blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:170]:
	layer.trainable = False

for layer in model.layers[170:]:
	layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
	loss='categorical_crossentropy',
	metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 resnet blocks)
# alongside the top Dense layers
print("Fine tune the model...")
model.fit_generator(train_generator, steps_per_epoch=1,
		validation_data = validation_generator)


##### Test the classifier

# Set the data generator for the testing dataset
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory('./test',
	target_size = (224, 224),
	batch_size = 32)
# Evaluates our model on the testing set and print accuracy and loss
print(model.metrics_names)
print(model.evaluate_generator(test_generator))

# TODO also call predict_generator to get the list of results

