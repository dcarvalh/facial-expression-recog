import os
import random
from shutil import copyfile

DATASET = "dataset_final"
TRAIN = "train"
VALID = "validation"
TEST = "test"
SETS = [TRAIN, VALID, TEST]


emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions

# Create folders
for sets in SETS:
    if not os.path.exists("{}".format(sets)):
        os.makedirs("{}".format(sets))

file_1 = open("{}{}{}.txt".format(TRAIN,os.sep,TRAIN),"w")
file_2 = open("{}{}{}.txt".format(VALID,os.sep,VALID),"w")
file_3 = open("{}{}{}.txt".format(TEST,os.sep,TEST),"w")

total_1 = 0
total_2 = 0
total_3 = 0

for emot in emotions:
    for sets in SETS:
        if not os.path.exists("{}{}{}".format(sets, os.sep, emot)):
            os.makedirs("{}{}{}".format(sets, os.sep, emot))

    # Divide Dataset

    images = os.listdir("{}{}{}".format(DATASET, os.sep, emot))

    images = list(images)

    random.shuffle(images)

    training = images[:int(len(images)*0.7)]
    validation = images[int(len(images)*0.7):int(len(images)*0.9)]
    testing = images[int(len(images)*0.9):]

    for image in training:
        copyfile("{}{}{}{}{}".format(DATASET, os.sep, emot, os.sep, image),
                 "{}{}{}{}{}".format(TRAIN, os.sep, emot, os.sep, image))
    file_1.write("{} - {}\n".format(len(training), emot))
    total_1 = total_1 + len(training)

    for image in validation:
        copyfile("{}{}{}{}{}".format(DATASET, os.sep, emot, os.sep, image),
                 "{}{}{}{}{}".format(VALID, os.sep, emot, os.sep, image))
    file_2.write("{} - {}\n".format(len(validation), emot))
    total_2 = total_2 + len(validation)

    for image in testing:
        copyfile("{}{}{}{}{}".format(DATASET, os.sep, emot, os.sep, image),
                 "{}{}{}{}{}".format(TEST, os.sep, emot, os.sep, image))
    file_3.write("{} - {}\n".format(len(testing), emot))
    total_3 = total_3 + len(testing)

file_1.write("\n{} - TOTAL\n".format(total_1))
file_2.write("\n{} - TOTAL\n".format(total_2))
file_3.write("\n{} - TOTAL\n".format(total_3))

file_1.close()
file_2.close()
file_3.close()