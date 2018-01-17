import os
from keras.preprocessing import image
import numpy as np
# Returns a numpy array based from an image file
# TODO fix error "obj has no attribute ndim"
# TODO default value of img_path temporary
def process_file(img_path="./train/anger/anger_10.jpg"):
	# load image
	img = image.load_img(img_path, target_size=(224, 224))
	# convert it to numpy array
	x = image.img_to_array(img)
	# add 1 more dimension (TODO why is it done ?)
	x = np.expand_dims(x, axis=0)
	# TODO what is this function ?
	# x = preprocess_input(x)
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
			img = process_file(os.path.join(path_e, img_name))
			# yield data and label
			yield (img, emotion)
