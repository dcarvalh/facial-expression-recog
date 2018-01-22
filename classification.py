# CAREFUL : not real finished working code. Just some quick writing in order to get familiar with Keras structure
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
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
# fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
# TODO adapt generate_tuples to our case

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2)	
train_generator = train_datagen.flow_from_directory('./train',
	target_size = (224, 224),
	batch_size = 32)

# TODO  samples_per_e = size(training set)/batch_size
model.fit_generator(train_generator, steps_per_epoch=4) #, epoch = 4)

##### Fine tuning

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from Resnet. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 resnet blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:170]:
	layer.trainable = False

for layer in model.layers[170:]:
	layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks)
# alongside the top Dense layers
model.fit_generator(generator, steps_per_epoch=10) #, epoch = 4)


############ Use the basic model without finetuning it 
# creates our CNN
model = ResNet50(weights='imagenet', classes=number_classes)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]



