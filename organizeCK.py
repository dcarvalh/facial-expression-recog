import os
from shutil import copyfile

SOURCE = "source_images"
LABELS = "source_labels"
OUTPUT = "sorted_set"

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] # Define emotion order

# check folders
for emot in emotions:
    if not os.path.exists("{}{}{}".format(OUTPUT, os.sep, emot)):
        os.makedirs("{}{}{}".format(OUTPUT, os.sep, emot))

# participants list
participants = os.listdir(LABELS)

for participant in participants:

    # participant's folders
    path_participant_folders = "{}{}{}".format(LABELS, os.sep, participant)
    participant_folders = os.listdir(path_participant_folders)

    save_neutral_flag = 1  # we want to save just ONE neutral image of each participant
    for folder in participant_folders:

        # LABELS
        # check corresponding emotion
        path_label = "{}{}{}{}{}".format(LABELS, os.sep, participant, os.sep, folder)
        label_file = os.listdir(path_label)


        if label_file == []:
            continue        # if no label, continue
        else:
            path_label = "{}{}{}".format(path_label, os.sep, label_file[0])

        fd = open(path_label, 'r')
        # read emotion number
        emotion_number = int(float(fd.readline()))  # read number

        # IMAGES
        path_images = "{}{}{}{}{}".format(SOURCE, os.sep, participant, os.sep, folder)
        participant_folder_images = os.listdir(path_images)
        participant_folder_images.sort() # orders images of one of the participants' folder

        # OUTPUT
        # copy first and last image
        first_file_path = "{}{}{}".format(path_images, os.sep, participant_folder_images[0])  # neutral
        last_file_path = "{}{}{}".format(path_images, os.sep, participant_folder_images[-1])   # emotion

        neutral_output_path = "{}{}{}{}{}".format(OUTPUT, os.sep, emotions[0], os.sep, participant_folder_images[0])
        emotion_output_path = "{}{}{}{}{}".format(OUTPUT, os.sep, emotions[emotion_number], os.sep,
                                                  participant_folder_images[-1])

        if save_neutral_flag == 1:
            copyfile(first_file_path, neutral_output_path)
            save_neutral_flag = 0
            
        copyfile(last_file_path, emotion_output_path)

print "\nSuccess!\n"