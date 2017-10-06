import tensorflow as tf
import os.path
import TrafficSignData
from NeuralNetwork import *
from NeuralNetworkOperations import *


class TrafficSignTrainer(object):
    def __init__(self, nn_graph, traffic_sign_data, epochs=10, batch_size=128, learning_rate=0.0005,
                 l2_regulizer_beta=0.001):
        self.file_name = self._get_timestamp_as_file_name()
        self.ts_data = traffic_sign_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, (None,
                                             ts_data.image_shape[0],
                                             ts_data.image_shape[1],
                                             ts_data.image_shape[2]))
        self.y = tf.placeholder(tf.int32, (None))
        self.nn = nn_graph(self.x, self.ts_data.n_classes)
        one_hot_y = tf.one_hot(self.y, self.ts_data.n_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=self.nn.get_logits())

        # Loss function using L2 Regularization
        loss_operation = tf.reduce_mean(cross_entropy + l2_regulizer_beta * self.nn.get_loss_regularizer())

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(self.nn.get_logits(), 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _get_timestamp_as_file_name(self):
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d-%H%M%S')

    def _evaluate(self):
        total_accuracy = 0
        sess = tf.get_default_session()

        for offset in range(0, self.ts_data.n_valid, self.batch_size):
            feed_dict = self.nn.get_feed_dict(training=False)
            feed_dict.update({self.x: self.ts_data.X_valid[offset:offset + self.batch_size],
                              self.y: self.ts_data.y_valid[offset:offset + self.batch_size]})
            accuracy = sess.run(self.accuracy_operation, feed_dict=feed_dict)
            total_accuracy += (accuracy * len(feed_dict[self.x]))
        return total_accuracy / self.ts_data.n_valid

    def train(self):
        from tqdm import tqdm
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print("Training...")
            print()
            pbar = tqdm(range(self.epochs))
            for i in pbar:
                self.ts_data.shuffle_training_data()
                for offset in range(0, self.ts_data.n_train, self.batch_size):
                    feed_dict = self.nn.get_feed_dict(training=True)
                    feed_dict.update({self.x: self.ts_data.X_train[offset:(offset + self.batch_size)],
                                      self.y: self.ts_data.y_train[offset:(offset + self.batch_size)]})
                    sess.run(self.training_operation, feed_dict=feed_dict)

                validation_accuracy = self._evaluate()
                pbar.set_description(f"Accuracy {validation_accuracy:.3}")

            file_path = os.path.join('./saved_models/', self.file_name)
            tf.train.Saver().save(sess, file_path)
            print("Model saved in ", file_path)


# Example:
# The TrafficSignTrainer with the LeNet graph described in the lesson, but here implemented using my
# tensorflow wrappers.

def LeNet(x, n_classes):
    nn = NeuralNetwork(x)
    nn.add(Conv2dOperation, n_output_channels=6, filter_shape=[5, 5])
    nn.add(MaxPoolOperation)
    nn.add(Conv2dOperation, n_output_channels=16, filter_shape=[5, 5])
    nn.add(MaxPoolOperation)
    nn.add(DenseOperation, n_output_channels=120)
    nn.add(DenseOperation, n_output_channels=84)
    nn.add(DenseOperation, n_output_channels=n_classes, activation=None)

    return nn


if __name__ == '__main__':
    ts_data = TrafficSignData.TrafficSignData()
    trainer = TrafficSignTrainer(nn_graph=LeNet, traffic_sign_data=ts_data, l2_regulizer_beta=0.0)
    trainer.train()
