from Inception import Inception
from DSHandler import DSHandler
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

number_classes = 4


# Will launch the classifier on one specific image
def test1image(model, img_path, dict_class):
    # Load the image
    img = image.load_img(img_path, target_size=(175, 175))
    # Change this image in numpy array to be suitable for our classifier
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the result from our classifier
    preds = model.predict(x)
    # Gives the class found and its confidence score
    print('Predicted : ', dict_class[preds.argmax()], 'with confidence of ', preds.max())


# Allow to loop through a float range
# Goes from start to stop with step step
def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step


#### Get all the useful variables and data
# Model definition
dsHandler = DSHandler((175, 175))
classifier = Inception(number_classes)

# load our model
classifier.model = load_model("Model_Exp3.h5")

# get generators for the testing
print("Evaluates classifier on testing set...")
test_gen = dsHandler.get_generator('./test')
classifier.evaluate(test_gen)

# Get dictionary of class label-indices (will be very useful later)
classes_dict = test_gen.class_indices
# Reverse the dict because we want it to be indexed by numbers and not class label
classes_dict = {v: k for k, v in classes_dict.iteritems()}

# Gives results for each sample as a list of probability
results = classifier.model.predict_generator(test_gen)

# Gives the output of our classifier
output_classif = results.argmax(axis=-1)

# Gives expected output for each sample
correct_classif = test_gen.classes


### Classify one image
test1image(classifier.model, "./test/neutral/neutral_10.jpg", classes_dict)


#### FAR/FRR Curve

# Number of false acceptance
nfa = 0
# Number of imposter attempts
nia = 0
# Number of failed rejection
nfr = 0
# number of legitimate access attemps
nea = 0

far = [0] * 11
frr = [0] * 11

i = 0
# Test for each threshold between 0 and 1
for thresh in frange(0, 1.0, 0.1):
	for result, expected in zip(results, correct_classif):
		# if the sample is well classified
		if result.argmax() == expected:
			nea = nea + 1
			# It its score was below our threshold
			if result.max() < thresh:
				nfr = nfr +1
		# If the sample isn't well classified
		else : 
			nia = nia +1
			# It its score was greater than our threshold
			if result.max() > thresh:
				nfa = nfa+1
	# Now that we have all values for this threshold, we can compute the 
	# corresponding FAR and FRR
	far[i] = float(nfa)/nia
	frr[i] = float(nfr)/nea
	# Resets values for the next threshold
	nea = nia = nfa = nfr = 0
	i = i+1

# Finally, we can prepare the plot for FAR and FRR
xi = [tr for tr in frange(0, 1.0, 0.1)]
far_plot = plt.plot(xi, far)
frr_plot = plt.plot(xi, frr)


#### Confusion Matrix
# Simple, just call the sklearn function once we have predicted and actual output
conf_matrix = confusion_matrix(correct_classif, output_classif)

print conf_matrix

print(correct_classif)
print(output_classif)


#### Build CMC Curve
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

# Get the rank of each element and sum it for our array
for e in correct_classif:
    rank = int(results[i][e])
    cmc[rank] = cmc[rank] + 1
    i = i + 1

# As the rank was inversed earlier, we need to reverse it again
cmc = cmc[::-1]
tot = sum(cmc)  # Should be equal to the number of samples

# Change values of cmc to get only the probabilities
cmc = [float(x) / tot for x in cmc]

# Computes the cumulative sum of those probabilities
cmc = [sum(cmc[:i]) for i in range(1, len(cmc) + 1)]


# Finally, we can plot our CMC
xi = [tr for tr in frange(0, 1.0, 0.1)]
plt.plot([x for x in range(1, 6)], cmc)
plt.show()
