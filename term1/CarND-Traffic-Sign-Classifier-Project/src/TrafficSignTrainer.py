import tensorflow as tf
import os.path
import difflib
import time

import TrafficSignData
from NeuralNetwork import *
from NeuralNetworkOperations import *
from NeuralNetworkGenerator import NeuralNetworkGenerator


class TrafficSignTrainer(object):
    def __init__(self, training_list, epochs=10, batch_size=128, learning_rate=0.0005,
                 l2_regulizer_beta=0.001, output_file_path="./saved_models/"):
        self.ts_data = TrafficSignData.TrafficSignData()
        self.nn_generator = NeuralNetworkGenerator(self.ts_data.n_classes, training_list, self.ts_data.image_shape)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_regulizer_beta = l2_regulizer_beta
        self.output_file_path = output_file_path
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)
        self.log_file_path = os.path.join(self.output_file_path, "log_file.csv")
        if not os.path.exists(self.log_file_path):
            self._write_header_to_log()
        self.prev_nn_graph_str_repr = ""  # Used to create diff printout with current graph

    def train(self):
        for nn_graph, self.x, self.y in self.nn_generator.generate():
            self._setup_training(nn_graph)
            self._run_training(nn_graph)

    def _setup_training(self, nn_graph):

        self._create_training_file_path()
        self._write_graph_to_file(nn_graph)
        one_hot_y = tf.one_hot(self.y, self.ts_data.n_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=nn_graph.get_logits())

        # Loss function using L2 Regularization
        loss_operation = tf.reduce_mean(cross_entropy + self.l2_regulizer_beta * nn_graph.get_loss_regularizer())

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(nn_graph.get_logits(), 1), tf.argmax(one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _run_training(self, nn_graph):
        from tqdm import tqdm
        highest_accuracy = -math.inf

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print(f"\nTraining id {self.current_training_unique_id}")
            self._print_diff(nn_graph)
            pbar = tqdm(range(self.epochs))
            self._timer_start()
            for i in pbar:
                self.ts_data.shuffle_training_data()
                for offset in range(0, self.ts_data.n_train, self.batch_size):
                    feed_dict = nn_graph.get_feed_dict(training=True)
                    feed_dict.update({self.x: self.ts_data.X_train[offset:(offset + self.batch_size)],
                                      self.y: self.ts_data.y_train[offset:(offset + self.batch_size)]})
                    sess.run(self.training_operation, feed_dict=feed_dict)

                accuracy = self._evaluate(nn_graph)
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    highest_accuracy_epoch = i

                pbar.set_description(f"Accuracy {accuracy:.03}({highest_accuracy:.03}@{highest_accuracy_epoch})")
            self._timer_stop()

            # Todo: save session when accuracy is highest
            tf.train.Saver().save(sess, os.path.join(self.current_training_file_path, "tensor_flow_model"))
            self._write_result_to_log(highest_accuracy, highest_accuracy_epoch)

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

    def _timer_start(self):
        self.start_time = (time.time(), time.process_time())

    def _timer_stop(self):
        end_time = (time.time(), time.process_time())
        self.wall_time = end_time[0] - self.start_time[0]
        self.cpu_time = end_time[1] - self.start_time[1]

    def _write_header_to_log(self):
        with open(self.log_file_path, 'w') as f:
            print(f"training-id, wall-time, cpu-time, highest accuracy, highest accuracy epoch,", file=f)

    def _write_result_to_log(self, highest_accuracy, highest_accuracy_epoch):
        with open(self.log_file_path, 'a') as f:
            print((f"{self.current_training_unique_id}, {self.wall_time:.01f}, {self.cpu_time:.01f}, "
                   f"{highest_accuracy:.04f}, {highest_accuracy_epoch},"), file=f)

    def _create_training_file_path(self):
        from datetime import datetime
        self.current_training_unique_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.current_training_file_path = os.path.join(self.output_file_path, self.current_training_unique_id)
        assert(os.path.exists(self.current_training_file_path) is False)
        os.makedirs(self.current_training_file_path)

    def _write_graph_to_file(self, nn_graph):
        with open(os.path.join(self.current_training_file_path, "NeuralNetworkOperations.txt"), 'w') as f:
            print(nn_graph, file=f)

    def _print_diff(self, nn_graph):
        """Print diff indicating what has been alternated from previous training.

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
    training_list = [
        Conv2dOperation.get_training_options(max_n_permutations=3, layer_size=0.0),
        MaxPoolOperation.get_training_options(max_n_permutations=1),
        Conv2dOperation.get_training_options(max_n_permutations=3, layer_size=0.2),
        MaxPoolOperation.get_training_options(max_n_permutations=1),
        DenseOperation.get_training_options(max_n_permutations=3, layer_size=0.05),
        DenseOperation.get_training_options(max_n_permutations=3, layer_size=0.01),
        DropoutOperation.get_training_options(max_n_permutations=1)
    ]

    trainer = TrafficSignTrainer(training_list=training_list, l2_regulizer_beta=0.0)
    trainer.train()
