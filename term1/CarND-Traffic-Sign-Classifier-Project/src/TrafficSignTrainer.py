import tensorflow as tf
import os.path
import difflib

import TrafficSignData
from NeuralNetwork import *
from NeuralNetworkOperations import *
from NeuralNetworkGenerator import NeuralNetworkGenerator


class TrafficSignTrainer(object):
    def __init__(self, nn_generator, traffic_sign_data, epochs=10, batch_size=128, learning_rate=0.0005,
                 l2_regulizer_beta=0.001):
        self.nn_generator = nn_generator
        self.ts_data = traffic_sign_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_regulizer_beta = l2_regulizer_beta
        self.prev_nn_graph_str_repr = ""

    def train(self):

        for nn_graph, self.x, self.y in self.nn_generator.generate():
            self._setup_training(nn_graph)
            self._run_training(nn_graph)

    def _setup_training(self, nn_graph):

        self.file_name = self._get_timestamp_for_file_name()
        one_hot_y = tf.one_hot(self.y, self.ts_data.n_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=nn_graph.get_logits())

        # Loss function using L2 Regularization
        loss_operation = tf.reduce_mean(cross_entropy + self.l2_regulizer_beta * nn_graph.get_loss_regularizer())

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(nn_graph.get_logits(), 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _get_timestamp_for_file_name(self):
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d-%H%M%S')

    def _run_training(self, nn_graph):
        from tqdm import tqdm
        import time

        start_time = (time.time(), time.process_time())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print("Training...")
            self._print_diff(nn_graph)
            pbar = tqdm(range(self.epochs))
            for i in pbar:
                self.ts_data.shuffle_training_data()
                for offset in range(0, self.ts_data.n_train, self.batch_size):
                    feed_dict = nn_graph.get_feed_dict(training=True)
                    feed_dict.update({self.x: self.ts_data.X_train[offset:(offset + self.batch_size)],
                                      self.y: self.ts_data.y_train[offset:(offset + self.batch_size)]})
                    sess.run(self.training_operation, feed_dict=feed_dict)

                validation_accuracy = self._evaluate(nn_graph)
                pbar.set_description(f"Accuracy {validation_accuracy:.3}")

            end_time = (time.time(), time.process_time())
            print(f"CPU time: {end_time[1] - start_time[1]}, Wall time: {end_time[0] - start_time[0]}")
            file_path = os.path.join('./saved_models/', self.file_name)
            tf.train.Saver().save(sess, file_path)
            print(f"Model saved as {file_path}")

    def _evaluate(self, nn_graph):
        total_accuracy = 0
        sess = tf.get_default_session()

        for offset in range(0, self.ts_data.n_valid, self.batch_size):
            feed_dict = nn_graph.get_feed_dict(training=False)
            feed_dict.update({self.x: self.ts_data.X_valid[offset:offset + self.batch_size],
                              self.y: self.ts_data.y_valid[offset:offset + self.batch_size]})
            accuracy = sess.run(self.accuracy_operation, feed_dict=feed_dict)
            total_accuracy += (accuracy * len(feed_dict[self.x]))
        return total_accuracy / self.ts_data.n_valid

    def _print_diff(self, nn_graph):
        """Print diff indicating what has been alternated from previous run.

        The graph from current training is printed, but things that's been alternated
        from previous training is printed in green.
        """
        current_str = nn_graph.__str__()
        prev_str = self.prev_nn_graph_str_repr
        match_list = difflib.SequenceMatcher(None, prev_str, current_str).get_matching_blocks()
        output_str = ""
        prev_match = (0, 0, 0)
        for current_match in match_list:
            if current_match[1] > prev_match[1]:
                # Add change as green text
                output_str += "\033[92m" + current_str[(prev_match[1] + prev_match[2]):current_match[1]] + "\033[0m"
            # Add equal text as default color
            output_str += current_str[current_match[1]: current_match[1] + current_match[2]]
            prev_match = current_match

        print(output_str)
        self.prev_nn_graph_str_repr = current_str


# Example:
# The TrafficSignTrainer with NeuralNetworkGenerator configured to produce variants of the LeNet graph described in
# the lesson.

if __name__ == '__main__':
    ts_data = TrafficSignData.TrafficSignData()
    training_list = [
        Conv2dOperation.get_training_options(max_n_tuning_permutations=3, layer_size=0.0),
        MaxPoolOperation.get_training_options(max_n_tuning_permutations=1),
        Conv2dOperation.get_training_options(max_n_tuning_permutations=3, layer_size=0.2),
        MaxPoolOperation.get_training_options(max_n_tuning_permutations=1),
        DenseOperation.get_training_options(max_n_tuning_permutations=3, layer_size=0.05),
        DenseOperation.get_training_options(max_n_tuning_permutations=3, layer_size=0.01),
        DropoutOperation.get_training_options(max_n_tuning_permutations=1)
    ]

    nn_generator = NeuralNetworkGenerator(ts_data.n_classes, training_list, ts_data.image_shape)
    trainer = TrafficSignTrainer(nn_generator=nn_generator, traffic_sign_data=ts_data, l2_regulizer_beta=0.0)
    trainer.train()
