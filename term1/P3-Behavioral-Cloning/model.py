import os
import csv
import cv2
import numpy as np
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('output_dir', './training_output/', "Output path for trained model and training information.")
flags.DEFINE_string('driving_log_dir', './data/example_data/', "Path to simulator driving log directories.")
flags.DEFINE_integer('epochs', 3, "Number of epochs to train")
flags.DEFINE_integer('batch_size', 32, "Batch size")


def load_driving_logs():
    """Loads all logs available in driving_log_dir.

    Returns:
        Dictionary of logs, where directory names are used as keys.
    """
    assert(os.path.exists(FLAGS.driving_log_dir))
    driving_logs = dict()
    for (dir_path, dir_names, file_names) in os.walk(FLAGS.driving_log_dir):
        if 'driving_log.csv' in file_names:
            name, log = load_driving_log(dir_path)
            driving_logs[name] = log
    return driving_logs


def load_driving_log(dir_path):
    """Load driving log located in selected directory.

    Returns:
        name -- head of dir_path
        log -- list of samples
    """
    file_path = os.path.join(dir_path, 'driving_log.csv')
    assert(os.path.exists(file_path))
    image_path = os.path.join(dir_path, 'IMG')
    assert(os.path.exists(image_path))

    _, name = os.path.split(dir_path)
    print(f"Loading {name}")
    log = []
    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file,
                                fieldnames=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
        for line in reader:
            log.append(fix_paths(line, image_path))

    return name, log


def fix_paths(csv_line, correct_path):
    """Correct file paths in read log data.

    The simulator outputs absolute paths to the log file, which becomes corrupt
    when the log is moved to another directory.
    """
    _, file_name = os.path.split(csv_line['center'])
    csv_line['center'] = os.path.join(correct_path, file_name)
    _, file_name = os.path.split(csv_line['left'])
    csv_line['left'] = os.path.join(correct_path, file_name)
    _, file_name = os.path.split(csv_line['right'])
    csv_line['right'] = os.path.join(correct_path, file_name)
    return csv_line


def combine_driving_logs(driving_logs):
    """Combine several driving logs into one

    Extracts a configurable number of samples from each log.

    Arguments:
        driving_logs -- Dictionary of logs, where directory names are used as keys.
    """
    combined_log = []
    for name, log in driving_logs.items():
        # For now use all available samples.
        combined_log += log
    return combined_log


def batch_generator(samples):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, FLAGS.batch_size):
            batch_samples = samples[offset:offset + FLAGS.batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample['center'])
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


def create_output_dir():
    """Generate an output dir for this training

    Current time-stamp is used to create a unique name for this training.

    Returns:
        Path to created directory
    """
    training_name = datetime.now().strftime('%Y%m%d-%H%M%S')
    training_dir = os.path.join(FLAGS.output_dir, training_name)
    assert(not os.path.exists(training_dir))
    os.makedirs(training_dir)
    print(f"Training output goes to {training_dir}")

    return training_dir


def plot_training_history_to_file(history_object, output_dir):
    """Plot training history

    Plot training and validation loss from each epoch

    Arguments:
        history_object -- provided by keras fit
        output_dir -- the figure is output as file
    """
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Mean Squared Error (MSE)')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(os.path.join(output_dir, 'training_result.png'), bbox_inches='tight')


def model_to_text_file(model, output_dir):
    try:
        with open(os.path.join(output_dir, 'model.txt'), 'w') as fid:
            model.summary(print_fn=lambda x: fid.write(x + '\n'))
    except TypeError:
        # Model summary print_fn wasn't implemented until Keras 2.0.6.
        print("Please update Keras library to >2.0.6 to get model summary written to file.")


def output_result(model, history_object):
    output_dir = create_output_dir()

    model.save(os.path.join(output_dir, 'model.h5'))
    plot_training_history_to_file(history_object, output_dir)
    model_to_text_file(model, output_dir)


def main(_):
    driving_logs = load_driving_logs()
    combined_log = combine_driving_logs(driving_logs)
    train_samples, validation_samples = train_test_split(combined_log, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = batch_generator(train_samples)
    validation_generator = batch_generator(validation_samples)

    model = define_model()
    model.compile(loss='mse', optimizer='adam')
#    model.fit_generator(train_generator, epochs=FLAGS.epochs,
#                        steps_per_epoch=(len(train_samples) // FLAGS.epochs),
#                        validation_data=validation_generator,
#                        validation_steps=(len(validation_samples) // FLAGS.epochs))
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator, nb_val_samples=len(validation_samples),
                                         nb_epoch=FLAGS.epochs, verbose=1)
    output_result(model, history_object)


if __name__ == '__main__':
    tf.app.run()
