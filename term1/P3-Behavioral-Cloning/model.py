import os
import csv
import cv2
import numpy as np
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('driving_log_dir', './data/data/', "Path to simulator driving log directory.")
flags.DEFINE_integer('epochs', 3, "Number of epochs to train")
flags.DEFINE_integer('batch_size', 32, "Batch size")


def load_driving_log():
    driving_log_file = os.path.join(FLAGS.driving_log_dir, 'driving_log.csv')
    assert(os.path.exists(driving_log_file))

    driving_log = []
    with open(driving_log_file) as csvfile:
        if csv.Sniffer.has_header:
            reader = csv.DictReader(csvfile)
        else:
            reader = csv.DictReader(csvfile,
                                    fieldnames=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

        for line in reader:
            driving_log.append(line)

    return driving_log


def batch_generator(samples):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, FLAGS.batch_size):
            batch_samples = samples[offset:offset + FLAGS.batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = batch_sample['center'].split('/')[-1]
                image_path = os.path.join(FLAGS.driving_log_dir, 'IMG', image_name)
                center_image = cv2.imread(image_path)
                center_angle = float(batch_sample['steering'])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def define_model(input_shape=(160, 320, 3)):
    """Defines the model to be trained.

    Implements something similar to the architecture described here:
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

    Returns:
        A Keras model
    """

    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))

    # Remove some parts of the sky/background (first 70 rows) and the vehicle hood (last 25 rows).
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Neural network layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model


def main(_):
    driving_log = load_driving_log()
    train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = batch_generator(train_samples)
    validation_generator = batch_generator(validation_samples)

    model = define_model()
    model.compile(loss='mse', optimizer='adam')
#    model.fit_generator(train_generator, epochs=FLAGS.epochs,
#                        steps_per_epoch=(len(train_samples) // FLAGS.epochs),
#                        validation_data=validation_generator,
#                        validation_steps=(len(validation_samples) // FLAGS.epochs))
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epochs)

    model.save('model.h5')


if __name__ == '__main__':
    tf.app.run()
