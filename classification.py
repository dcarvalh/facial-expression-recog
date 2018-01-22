
from ResNet import ResNet
from DSHandler import DSHandler

number_classes = 7

##### Model definition
dsHandler = DSHandler((197,197))
classifier = ResNet(number_classes)

# get generators for the training and validation set
train_gen = dsHandler.get_train_generator('./train')
val_gen = dsHandler.get_generator('./validation')

print("First training...")
classifier.first_train(train_gen, val_gen)

print("Fine tune the model...")
classifier.fine_tune(train_gen, val_gen)

# get generators for the testing
#print("Evaluates classifier on testing set...")
#test_gen = dsHandler.get_generator('./test')
#classifier.evaluate(test_gen)





