from Inception import Inception
from DSHandler import DSHandler
from keras.models import load_model


number_classes = 7


##### Model definition
dsHandler = DSHandler((175,175))
classifier = Inception(number_classes)



classifier.model = load_model("MyModel.h5")

# get generators for the testing
print("Evaluates classifier on testing set...")
test_gen = dsHandler.get_generator('./test')
classifier.evaluate(test_gen)

# Confusion matrix
print("\nConfusion Matrix")
print("\n#Labels#")
classes_dict = test_gen.class_indices
classes_view = [(v, k)for k, v in classes_dict.iteritems()]
classes_view.sort() # natively sort tuples by first element
for v, k in classes_view:
    print "%d - %s" % (v, k)

classifier.conf_matrix(test_gen)





