from Inception import Inception
from DSHandler import DSHandler
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

number_classes = 6

# Model definition
dsHandler = DSHandler((175, 175))
classifier = Inception(number_classes)

classifier.model = load_model("MyModel.h5")

# get generators for the testing
print("Evaluates classifier on testing set...")
test_gen = dsHandler.get_generator('./test')
classifier.evaluate(test_gen)

# Confusion matrix
print("\nConfusion Matrix")
print("\n#Labels#")
# Get dictionary of class label-indices
classes_dict = test_gen.class_indices
# Build a list containing dictionary values sorted
sorted_classes = sorted(classes_dict.items(), key=operator.itemgetter(1))
for clas, val in sorted_classes:
    print "%d - %s" % (val, clas)

classifier.conf_matrix(test_gen)

#### Build CMC Curve
# gives results for each sample as a list of probability
results = classifier.model.predict_generator(test_gen)

i = 0
# Build a new array containing only ranks
for r in results:
	temp = r.argsort()
	# returns new array with same shape & type as temp
	ranks = np.empty_like(temp)
	# Returns range of evenly spaced value between 0 and len(r)
	ranks[temp] = np.arange(len(r))
	# update our array with those ranks
	results[i]=ranks
	i = i+1

i = 0
cmc = [0, 0, 0, 0, 0, 0]

# gives expected output for each sample
expected = test_gen.classes
# Get the rank of each element and sum it for our array
# TODO find a simpler way to do it
for e in expected:
	rank = int(results[i][e])
	cmc[rank] = cmc[rank] + 1	
	i=i+1


# As the rank was inverse earlier, we need to reverse it again
cmc = cmc[::-1]
print(sum(cmc)) # Should be equal to the number of samples

# change values of cmc to get only the probabilities
cmc = [float(x) / 39. for x in cmc]

print(sum(cmc)) # Should be (almost) equal to 1

# Computes the cumulative sum of those probabilities
cmc = [sum(cmc[:i]) for i in range(1, len(cmc)+1)]

# Finally, we can plot our CMC
plt.plot(cmc)
plt.show()

