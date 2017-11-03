import os
import sys
import csv
import cv2
import numpy as np
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
import enum
import itertools

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('output_dir', './training_output/', "Output path for trained model and training information.")
flags.DEFINE_string('driving_log_dir', './data/example_data/', "Path to simulator driving log directories.")
flags.DEFINE_integer('epochs', 3, "Number of epochs to train")
flags.DEFINE_integer('batch_size', 32, "Batch size")
flags.DEFINE_float('steering_offset', 0.0245, "Steering offset for left and right camera images.")
flags.DEFINE_float('steering_equality_factor', 0.3, ("Equalization factor for angle distribution in the training set. ",
                                                     "This value should be in the range [0, 1], where 0 gives equal ",
                                                     "distribution, and 1 keeps the distribution imbalanced."))
flags.DEFINE_float('sharp_turn_threshold', 0.05, ("A steering value greater than this during normal driving is",
                                                  "assumed to be in a sharp curve."))


class SequenceType(enum.Enum):
    NO_LANE = 1,
    RIGHT_LANE = 2,
    LEFT_LANE = 3,
    SHADOWS = 4


class CamType(enum.Enum):
    CENTER = 1,
    LEFT = 2,
    RIGHT = 3,


def load_driving_logs():
    """Loads all logs available in driving_log_dir.

    Returns:
        Dictionary of logs, where directory names are used as keys.
    """
    assert(os.path.exists(FLAGS.driving_log_dir))
    driving_logs = dict()
    for (dir_path, dir_names, file_names) in os.walk(FLAGS.driving_log_dir):
        if 'driving_log.csv' in file_names:
            sequence_type, log = load_driving_log(dir_path)
            if sequence_type in driving_logs.keys():
                driving_logs[sequence_type] += log
            else:
                driving_logs[sequence_type] = log
    return driving_logs


def load_driving_log(dir_path):
    """Load driving log located in selected directory.

    Returns:
        sequence_type -- defined by dir name
        log -- list of samples
    """
    file_path = os.path.join(dir_path, 'driving_log.csv')
    assert(os.path.exists(file_path))
    image_path = os.path.join(dir_path, 'IMG')
    assert(os.path.exists(image_path))

    _, name = os.path.split(dir_path)
    sequence_type = dir_name_to_sequence_type(name)
    print(f"Loading {name} as {sequence_type}")
    log = []
    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file,
                                fieldnames=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
        for line in reader:
            log.append(fix_paths(line, image_path))

    return sequence_type, log


def dir_name_to_sequence_type(name):
    # Map directory names to SequenceType
    sequence_type_lookup = {
        'no_lane_cw': SequenceType.NO_LANE,
        'no_lane_ccw': SequenceType.NO_LANE,
        'no_lane_ccw_recovery': SequenceType.NO_LANE,
        'right_lane_cw': SequenceType.RIGHT_LANE,
        'right_lane_ccw': SequenceType.RIGHT_LANE,
        'right_lane_cw_recovery': SequenceType.RIGHT_LANE,
        'right_lane_ccw_recovery': SequenceType.RIGHT_LANE,
        'left_lane_cw': SequenceType.LEFT_LANE,
        'left_lane_ccw': SequenceType.LEFT_LANE,
        'left_lane_cw_recovery': SequenceType.LEFT_LANE,
        'left_lane_ccw_recovery': SequenceType.LEFT_LANE,
        'righ_lane_shadows': SequenceType.SHADOWS,
    }
    try:
        sequence_type = sequence_type_lookup[name]
    except KeyError:
        print(f"Please name driving log directory {name} one of {list(sequence_type_lookup.keys())}")
        sys.exit(1)

    return sequence_type


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


Sample = namedtuple('Sample', ['image_path', 'steering', 'sequence_type', 'cam_type', 'flip', 'augment'])


def driving_logs_to_samples(driving_logs):
    """Combine several driving logs into one list of samples

    Arguments:
        driving_logs -- Dictionary of logs, where directory names are used as keys.
    """
    all_samples = []
    for sequence_type, log in driving_logs.items():
        samples = log_to_samples(log, sequence_type)
        print(sequence_type, len(samples))
        all_samples += samples

    return all_samples


def log_to_samples(log, sequence_type):
    samples = []
    for log_entry in log:
        cam_variants = [[('center', CamType.CENTER), ('left', CamType.LEFT), ('right', CamType.RIGHT)]]
        flip_variants = [[False, True]]
        augment_variants = [[False, True]]
        for cam, flip, augment in itertools.product(*(cam_variants + flip_variants + augment_variants)):
            samples.append(Sample(image_path=log_entry[cam[0]],
                                  steering=float(log_entry['steering']),
                                  sequence_type=sequence_type,
                                  cam_type=cam[1],
                                  flip=flip,
                                  augment=augment))

    return samples


def oversample_shadows(samples):
    """Oversample sequence type shadow"""

    # Oversample every shadow example by this count.
    oversample_count = 5

    extra_samples = []
    for sample in samples:
        if sample.sequence_type is SequenceType.SHADOWS:
            for i in range(oversample_count):
                extra_samples.append(sample)
    samples += extra_samples
    return samples


def filter_samples(samples):
    """Remove samples causing worse result.

    Arguments:
        samples -- List of samples

    Returns:
        samples -- Filtered list of samples
    """

    # Use left and right camera only to stabilize when going fairly straight.
    samples = list(filter(lambda sample:
                          (sample.cam_type is CamType.CENTER or
                           abs(sample.steering) < FLAGS.sharp_turn_threshold),
                          samples))

    # The model being trained shall drive in the right lane or just centered on the road when there is no lane. So left
    # lane sequences are only to be used with image flip, and right lane sequences without flip. No-lane sequences are
    # used both w/ and w/o flip.
    samples = list(filter(lambda sample:
                          ((sample.sequence_type in [SequenceType.LEFT_LANE, SequenceType.NO_LANE] and
                            sample.flip) or
                           (sample.sequence_type is not SequenceType.LEFT_LANE and
                            not sample.flip)),
                          samples))

    # Augment track 1 with shadows. To avoid stacking shadows, this can't be done on track 2 (without a shadow
    # detector).
    samples = list(filter(lambda sample:
                          ((sample.sequence_type is SequenceType.NO_LANE and sample.augment) or not sample.augment),
                          samples))

    return samples


def adjust_steering(samples):
    """Adjust steering angles

    Apply a steering offset to left and right camera samples towards center.

    Arguments:
        samples -- List of samples to be adjusted.

    Returns:
        samples -- Adjusted list of samples
    """
    for i in range(len(samples)):
        # Adjust for left and right cam
        steering = samples[i].steering
        if samples[i].cam_type is CamType.LEFT:
            samples[i] = samples[i]._replace(steering=steering + FLAGS.steering_offset)
        elif samples[i].cam_type is CamType.RIGHT:
            samples[i] = samples[i]._replace(steering=steering - FLAGS.steering_offset)
        else:
            # Do nothing for center camera.
            pass

        # Adjust for horizontal flip
        steering = samples[i].steering
        if samples[i].flip:
            samples[i] = samples[i]._replace(steering=-steering)

    return samples


def equalize_samples(samples):
    """Equalize the angles in the training set by over and under sampling.

    The equality of output angles is controlled by the application argument
    steering_equality_factor. This value should be in range [0, 1], where
    0 makes it equal, and 1 keeps the imbalanced distribution.
    """
    angles = np.array([sample.steering for sample in samples[:]])
    abs_max = np.max(np.abs(angles))
    input_hist, bin_edges = np.histogram(angles, bins=21, range=(-abs_max, abs_max))
    bin_edges[-1] += 0.00001

    # Calculate the output histogram, which should be the result after over and under sampling the input samples.
    # Large bins are reduced and small ones increased to achieve equality as regulated by steering_equality_factor.
    output_hist = (input_hist - input_hist.mean()) * FLAGS.steering_equality_factor + input_hist.mean()
    keep_rates = output_hist / input_hist

    samples_out = []
    for sample in samples:
        keep_rate = keep_rates[np.digitize(sample.steering, bin_edges) - 1]
        if keep_rate < 1.0:
            if keep_rate > np.random.random():
                count = 1
            else:
                count = 0
        else:
            count = int(np.round(keep_rate * (np.random.random() + 0.5)))
        for i in range(count):
            samples_out.append(sample)

    angles_out = np.array([sample.steering for sample in samples_out[:]])
    plt.figure()
    plt.hist(angles, bins=21, alpha=0.5, label='in')
    plt.hist(angles_out, bins=21, alpha=0.5, label='out')
    plt.legend(loc='upper right')
    plt.show()

    return samples_out


def augment_shadow(bgr_image):
    """Augment a shadow into the image.

    A shadow is randomly generated and drawn into the image. This by dividing
    the image into two parts by a straight line, where one side is darkened.
    The darkening is done without affecting the colors very much, which is
    achieved by first converting color-space from BGR to HSV, and then only
    modify the V-channel. The image is then converted back to BGR.

    Arguments:
        bgr_image - The BGR image to be augmented with a shadow.

    Returns:
        bgr_image - The augmented BGR image.
    """

    # This is an attempt to reverse engineer the shadows in track 2. The gray-scale value from different materials have
    # been sampled both in shadow and sunlight. A scatter plot showed that the measurements roughly are on a straight
    # line intercepting (0, 0) which is reasonable since black should be black also when applying a shadow. The
    # straight line is then estimated by the mean ratio from all measurements.
    shadow_ratios = {'lane': 51 / 160, 'pavement': 40 / 125, 'grass': 20 / 60, 'dark_cliff': 5 / 22}
    shadow_factor = np.mean([value for value in shadow_ratios.values()])
    shadow_types = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}
    height, width, _ = bgr_image.shape

    # Convert to HSV to easily add shadow without changing color.
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Randomly select which side to apply shadow on.
    shadow_type = shadow_types[np.floor(np.random.random() * len(shadow_types)).astype(int)]

    # Two points are randomly generated on the image border, and one side is then darkened by applying the shadow
    # factor on the V-channel.
    if shadow_type is 'left':
        x_top, x_bottom = np.floor(np.random.random(2) * width).astype(int)
        delta = (x_bottom - x_top) / height
        for y_idx in range(height):
            hsv_image[y_idx, :(x_top + np.round(delta * y_idx).astype(np.int)), 2] = \
                (hsv_image[y_idx, :(x_top + np.round(delta * y_idx).astype(np.int)), 2] *
                 shadow_factor).astype(np.int8)
    elif shadow_type is 'right':
        x_top, x_bottom = np.floor(np.random.random(2) * width).astype(int)
        delta = (x_bottom - x_top) / height
        for y_idx in range(height):
            hsv_image[y_idx, (x_top + np.round(delta * y_idx).astype(np.int)):, 2] = \
                (hsv_image[y_idx, (x_top + np.round(delta * y_idx).astype(np.int)):, 2] *
                 shadow_factor).astype(np.int8)
    elif shadow_type is 'bottom':
        y_left, y_right = np.floor(np.random.random(2) * height).astype(int)
        delta = (y_right - y_left) / width
        for x_idx in range(width):
            hsv_image[(y_left + np.round(delta * x_idx).astype(np.int)):, x_idx, 2] = \
                (hsv_image[(y_left + np.round(delta * x_idx).astype(np.int)):, x_idx, 2] *
                 shadow_factor).astype(np.int8)
    elif shadow_type is 'top':
        y_left, y_right = np.floor(np.random.random(2) * height).astype(int)
        delta = (y_right - y_left) / width
        for x_idx in range(width):
            hsv_image[:(y_left + np.round(delta * x_idx).astype(np.int)), x_idx, 2] = \
                (hsv_image[:(y_left + np.round(delta * x_idx).astype(np.int)), x_idx, 2] *
                 shadow_factor).astype(np.int8)

    # Convert back the augmented image to BGR.
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return bgr_image


def batch_generator(samples):
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, len(samples), FLAGS.batch_size):
            batch_samples = samples[offset:offset + FLAGS.batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                image = cv2.imread(sample.image_path)
                if sample.flip:
                    image = np.fliplr(image)
                if sample.augment:
                    image = augment_shadow(image)

                images.append(image)
                angles.append(sample.steering)

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
    model.add(Dropout(rate=0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate=0.1))
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


def setup_callbacks_list(output_path):
    """Setup a list of training callbacks"""
    file_path = os.path.join(output_path, "weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5")
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    return [checkpoint]  # Currently only one callback


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

    with open(os.path.join(output_dir, 'model.txt'), 'a') as fid:
        print("\nArguments:", file=fid)
        for key, value in FLAGS.__flags.items():
            print(key, value, file=fid)


def output_result(output_path, model, history_object):
    plot_training_history_to_file(history_object, output_path)
    model_to_text_file(model, output_path)


def main(_):
    output_path = create_output_dir()
    driving_logs = load_driving_logs()
    samples = driving_logs_to_samples(driving_logs)
    samples = filter_samples(samples)
    samples = adjust_steering(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_samples = oversample_shadows(train_samples)
    train_samples = equalize_samples(train_samples)
    train_generator = batch_generator(train_samples)
    validation_generator = batch_generator(validation_samples)

    model = define_model()
    model.compile(loss='mse', optimizer='adam')

    callbacks_list = setup_callbacks_list(output_path)
    history_object = model.fit_generator(train_generator, epochs=FLAGS.epochs,
                                         steps_per_epoch=(len(train_samples) // FLAGS.batch_size),
                                         validation_data=validation_generator,
                                         validation_steps=(len(validation_samples) // FLAGS.batch_size),
                                         callbacks=callbacks_list, verbose=1)
    output_result(output_path, model, history_object)


if __name__ == '__main__':
    tf.app.run()
