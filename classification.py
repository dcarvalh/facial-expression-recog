from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

from ResNet import ResNet

number_classes = 7

##### Model definition
classifier = ResNet(number_classes)

print("First training...")
classifier.first_train()

print("Fine tune the model...")
classifier.fine_tune()

print("Evaluates classifier on testing set...")
classifier.evaluate()





