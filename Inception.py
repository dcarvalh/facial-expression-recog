from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix


class Inception:
    def __init__(self, number_classes):
        # create the base pre-trained model
        self.base_model = InceptionV3(weights='imagenet', include_top=False)

        # Build a classifier model to put on top
        # add a global spatial average pooling layer
        layer = self.base_model.output
        layer = GlobalAveragePooling2D()(layer)
        # Add a fully-connected layer
        layer = Dense(1024, activation='relu')(layer)
        # and a logistic layer for our classes
        predict = Dense(number_classes, activation='softmax')(layer)

        # this is the model we will train
        self.model = Model(inputs=self.base_model.input, outputs=predict)

    ##### First training
    ## Do a first training of the model on the new data for a few epoch
    def first_train(self, train_gen, val_gen):
        # first: train only new layers, which were randomly initialized
        # i.e. freeze all other ResNet layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # compile the model
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # steps_per_epoch is number of samples divided by batch size
        self.model.fit_generator(train_gen,
                                 steps_per_epoch=len(train_gen.classes) // train_gen.batch_size,
                                 epochs=4,
                                 validation_data=val_gen,
                                 validation_steps=len(val_gen.classes) // val_gen.batch_size)

    ##### Fine tuning
    # Here, the top layers are trained and we can start fine-tuning
    # convolutional layers from Resnet. Freeze the bottom N layers
    # and train the remaining top layers.
    def fine_tune(self, train_gen, val_gen):
        # Chose to train the top 2 inception blocks,
        # i.e., freeze the bottom N layers and unfreeze the rest:
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        # Need to recompile the model for these changes to take effect
        # As recommended, SGD is used with a low learning rate
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # Train model again (now fine-tuning the top 2 inception blocks)
        # alongside the top Dense layers
        self.model.fit_generator(train_gen,
                                 steps_per_epoch=len(train_gen.classes) // train_gen.batch_size,
                                 epochs=4,
                                 validation_data=val_gen,
                                 validation_steps=len(val_gen.classes) // val_gen.batch_size)

    # Evaluates our classifier on the testing set and print accuracy and loss
    def evaluate(self, test_gen):
        print(self.model.metrics_names)
        print(self.model.evaluate_generator(test_gen))


    #### Generates the confusion matrix
    def conf_matrix(self, test_gen):
        # expected output
        correct_classif = test_gen.classes
	# our classifier output
        output_classif = self.model.predict_generator(test_gen).argmax(axis=-1)
        print(correct_classif)
        print(output_classif)

	# computes the confusion matrix
        conf_matrix = confusion_matrix(correct_classif, output_classif)

        # Print the confusion matrix
        print conf_matrix
