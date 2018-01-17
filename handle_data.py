import os
import numpy as np
# Returns a numpy array based from an image file
def process_file(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x


# Returns a generator
def generate_tuples(path):
	#while 1:
	# Get the list of all classes (= list of folder in the dataset path)
	classes = [dr for dr in os.listdir(path) if os.path.isdir(os.path.join(path, dr))]
	# For each class...
	for emotion in classes:
		path_e = os.path.join(path, emotion)
		# For each image file in this directory...
		for img_name in os.listdir(path_e):
			# create Numpy arrays of input data
			img = process_line(os.path.join(path_e, img_name))
			# yield data and label
			yield (data, emotion)
