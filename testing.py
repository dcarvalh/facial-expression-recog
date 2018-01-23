from Inception import Inception
from DSHandler import DSHandler
from keras.models import load_model

number_classes = 7

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
classes_dict = test_gen.class_indices
classes_view = [(v, k)for k, v in classes_dict.iteritems()]
classes_view.sort()  # natively sort tuples by first element
for v, k in classes_view:
    print "%d - %s" % (v, k)

classifier.conf_matrix(test_gen)

#### CMC
# gives expected output for each sample
expected = test_gen.classes
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
	results[i]=ranks
	i = i+1

i = 0
cmc = [0, 0, 0, 0, 0, 0]
# Get the rank of each element and sum it for our array
for e in expected:
	rank = int(results[i][e])
	cmc[rank] = cmc[rank] + 1	
	i=i+1

# Should be equal to the number of samples
print(sum(cmc))
# change values of cmc to get only the probabilities
cmc = [float(x) / 39. for x in cmc]
# Should be (at least almost) equal to 1
print(sum(cmc))
# Computes the cumulative sum of those probabilities
cmc = [sum(cmc[:i]) for i in range(1, len(cmc)+1)]

# Finally, we can plot our CMC
plt.plot(cmc)
plt.show()

