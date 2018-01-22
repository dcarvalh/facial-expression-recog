
from ResNet import ResNet
from DSHandler import DSHandler

number_classes = 7

##### Model definition
dsHandler = DSHandler()
classifier = ResNet(number_classes)

print("First training...")
classifier.first_train(dsHandler.train_generator, dsHandler.validation_generator)

print("Fine tune the model...")
classifier.fine_tune(dsHandler.train_generator, dsHandler.validation_generator)
#
#print("Evaluates classifier on testing set...")
#classifier.evaluate(dsHandler.test_generator)





