from Inception import Inception
from DSHandler import DSHandler
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

number_classes = 4


def test1image(model, img_path, dict_class):
    img = image.load_img(img_path, target_size=(175, 175))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability
    # (one such list for each sample in the batch)
    print('Predicted : ', dict_class[preds.argmax()], 'with confidence of ', preds.max())


# Model definition
dsHandler = DSHandler((175, 175))
classifier = Inception(number_classes)

# load our model
classifier.model = load_model("MyModel_new_48.h5")

# get generators for the testing
print("Evaluates classifier on testing set...")
test_gen = dsHandler.get_generator('./test')
classifier.evaluate(test_gen)

# Get dictionary of class label-indices (will be very useful later)
classes_dict = test_gen.class_indices
# The dictionary found is not suitable. 
# We want it to be indexed by index and not class label so it's reversed
classes_dict = {v: k for k, v in classes_dict.iteritems()}

### Try to classify one image
test1image(classifier.model, "./test/neutral/neutral_10.jpg", classes_dict)

#### Build CMC Curve
# gives results for each sample as a list of probability
results = classifier.model.predict_generator(test_gen)

correct_classif = test_gen.classes
output_classif = results.argmax(axis=-1)
conf_matrix = confusion_matrix(correct_classif, output_classif)

print conf_matrix

print(correct_classif)
print(output_classif)


i = 0
# Build a new array containing only ranks
for r in results:
    temp = r.argsort()
    # returns new array with same shape & type as temp
    ranks = np.empty_like(temp)
    # Returns range of evenly spaced value between 0 and len(r)
    ranks[temp] = np.arange(len(r))
    # update our array with those ranks
    results[i] = ranks
    i = i + 1

i = 0
cmc = number_classes * [0]


# gives expected output for each sample
expected = test_gen.classes
# Get the rank of each element and sum it for our array
for e in expected:
    rank = int(results[i][e])
    cmc[rank] = cmc[rank] + 1
    i = i + 1

# As the rank was inverse earlier, we need to reverse it again
cmc = cmc[::-1]
tot = sum(cmc)  # Should be equal to the number of samples

# change values of cmc to get only the probabilities
cmc = [float(x) / tot for x in cmc]

# Computes the cumulative sum of those probabilities
cmc = [sum(cmc[:i]) for i in range(1, len(cmc) + 1)]

# Finally, we can plot our CMC
plt.plot(cmc)
plt.show()
