import cv2
import glob
import os

SOURCE = "source_final"
OUTPUT = "dataset_final"

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotions


# check folders
if not os.path.exists("{}".format(OUTPUT)):
    os.makedirs("{}".format(OUTPUT))

for emot in emotions:
    if not os.path.exists("{}{}{}".format(OUTPUT, os.sep, emot)):
        os.makedirs("{}{}{}".format(OUTPUT, os.sep, emot))

faceClass_1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceClass_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceClass_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceClass_4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

for emot in emotions:

    files = glob.glob("{}{}{}{}*".format(SOURCE, os.sep, emot, os.sep))  # all paths of images in a specific emotion

    num = 0

    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        # Detect a face using 4 different classifiers in cascade
        face = faceClass_1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face) != 1:      # if a face is not detected try another recognizer
            face = faceClass_2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face) != 1:      # if a face is not detected try another recognizer
                face = faceClass_3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
                if len(face) != 1:      # if a face is not detected try another recognizer
                    face = faceClass_4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(face) != 1:      # if a face is not detected print error
                        print "ERROR face was not recongized in file {}!".format(f)
                        face = ""

        # Cut and save face
        for (x, y, w, h) in face:  # get coordinates and size of rectangle containing face

            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite("{}{}{}{}{}_{}.jpg".format(OUTPUT, os.sep, emot, os.sep, emot, num), out)
            except:
                print "ERROR file {} was not written!".format(f)     # If error, print it to the screen

        num += 1  # file number increment
