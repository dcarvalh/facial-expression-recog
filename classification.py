from Inception import Inception
from DSHandler import DSHandler

number_classes = 4

# Model definition
dsHandler = DSHandler((175, 175))
classifier = Inception(number_classes)

# get generators for the training and validation set
train_gen = dsHandler.get_train_generator('./train')
val_gen = dsHandler.get_generator('./validation')

print("First training...")
classifier.first_train(train_gen, val_gen)

print("Fine tune the model...")
classifier.fine_tune(train_gen, val_gen)

# get generators for the testing
print("Evaluates classifier on testing set...")
test_gen = dsHandler.get_generator('./test')

# TODO all this part (except save model) useful ?
print("\nConfusion Matrix")
print("\n#Labels#")
classes_dict = test_gen.class_indices
classes_view = [(v, k)for k, v in classes_dict.iteritems()]
classes_view.sort()  # natively sort tuples by first element
for v, k in classes_view:
    print "%d - %s" % (v, k)


classifier.evaluate(test_gen)

my_model = classifier.model

# plot_model(my_model, to_file='model.png')

# Save model learned in a file to get statistics on this model
my_model.save("MyModel_b1.h5")

