import tensorflow as tf
import os.path
import difflib
import time
import pandas as pd

import TrafficSignData
from NeuralNetwork import *
from NeuralNetworkOperations import *
from NeuralNetworkGenerator import NeuralNetworkGenerator


class TrafficSignTrainer(object):
    def __init__(self, training_list, epochs=10, batch_size=128, learning_rate=0.0005,
                 l2_regulizer_beta=0.001, project_file_path="./"):
        self.ts_data = TrafficSignData.TrafficSignData()
        self.nn_generator = NeuralNetworkGenerator(self.ts_data.n_classes, training_list, self.ts_data.image_shape)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2_regulizer_beta = l2_regulizer_beta
        self.project_file_path = project_file_path
        self.output_file_path = os.path.join(project_file_path, "./training_output/")
        if not os.path.exists(self.output_file_path):
            os.makedirs(self.output_file_path)
        self.log_file_path = os.path.join(self.output_file_path, "log_file.csv")
        if not os.path.exists(self.log_file_path):
            self._write_header_to_log()
        self.prev_nn_graph_str_repr = ""  # Used to create diff printout with current graph

    def train(self):
        for idx, (nn_graph, self.x, self.y) in enumerate(self.nn_generator.generate()):
            self._setup_training(nn_graph)
            print(f"\nTraining ({idx + 1}/{self.nn_generator.get_n_trainings()}): {self.training_unique_id}")
            self._print_diff(nn_graph)
            self._run_training(nn_graph)
            self._calc_training_stat()
            self._write_training_stat_to_log()

    def _setup_training(self, nn_graph):
        self.training_stat = {'accuracy': -math.inf}
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

        self.confusion_matrix_operation = tf.confusion_matrix(tf.argmax(one_hot_y, 1),
                                                              tf.argmax(nn_graph.get_logits(), 1),
                                                              self.ts_data.n_classes)

    def _evaluate(self, nn_graph):
        total_accuracy = 0
        total_confusion_matrix = np.zeros((self.ts_data.n_classes, self.ts_data.n_classes))
        sess = tf.get_default_session()

        for offset in range(0, self.ts_data.n_valid, self.batch_size):
            feed_dict = nn_graph.get_feed_dict(training=False)
            feed_dict.update({self.x: self.ts_data.X_valid[offset:offset + self.batch_size],
                              self.y: self.ts_data.y_valid[offset:offset + self.batch_size]})
            accuracy, confusion_matrix = sess.run([self.accuracy_operation, self.confusion_matrix_operation],
                                                  feed_dict=feed_dict)
            total_accuracy += (accuracy * len(feed_dict[self.x]))
            total_confusion_matrix += confusion_matrix

        return total_accuracy / self.ts_data.n_valid, total_confusion_matrix

    def _run_training(self, nn_graph):
        from tqdm import tqdm
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pbar = tqdm(range(self.epochs))
            self._timer_start()
            for i in pbar:
                self.ts_data.shuffle_training_data()
                for offset in range(0, self.ts_data.n_train, self.batch_size):
                    feed_dict = nn_graph.get_feed_dict(training=True)
                    feed_dict.update({self.x: self.ts_data.X_train[offset:(offset + self.batch_size)],
                                      self.y: self.ts_data.y_train[offset:(offset + self.batch_size)]})
                    sess.run(self.training_operation, feed_dict=feed_dict)

                accuracy, confusion_matrix = self._evaluate(nn_graph)
                if accuracy > self.training_stat['accuracy']:
                    self.training_stat['accuracy'] = accuracy
                    self.training_stat['epochs'] = i
                    self.training_stat['confusion_matrix'] = confusion_matrix
                    tf.train.Saver().save(sess, os.path.join(self.training_file_path, "tensor_flow_model"))

                pbar.set_description(f"Accuracy {accuracy:.03} ({self.training_stat['accuracy']:.03}@"
                                     f"{self.training_stat['epochs']})")
            self._timer_stop()

    def _timer_start(self):
        self.start_time = (time.time(), time.process_time())

    def _timer_stop(self):
        end_time = (time.time(), time.process_time())
        self.training_stat['wall_time'] = end_time[0] - self.start_time[0]
        self.training_stat['cpu_time'] = end_time[1] - self.start_time[1]

    def _write_header_to_log(self):
        with open(self.log_file_path, 'w') as f:
            print(("training-id,wall-time,cpu-time,epoch,accuracy,lowest precision,lowest sensitivity,"
                   "lowest specificity,lowest f1 score"), file=f)

    def _calc_training_stat(self):
        """Calculate some additional statistics based on the confusion matrix"""

        # Num predicted detections per class (sum cols)
        predicted = self.training_stat['confusion_matrix'].sum(axis=0)

        # Num expected detections per class (sum rows)
        expected = self.training_stat['confusion_matrix'].sum(axis=1)

        # Calculate metrics per class/label
        true_positives = np.diag(self.training_stat['confusion_matrix'])
        false_positives = expected - true_positives
        false_negatives = predicted - true_positives
        true_negatives = (expected.sum() - true_positives - false_positives - false_negatives)

        self.training_stat['precision'] = true_positives / (true_positives + false_positives)
        self.training_stat['sensitivity'] = true_positives / (true_positives + false_negatives)
        self.training_stat['specificity'] = true_negatives / (true_negatives + false_positives)
        self.training_stat['f1_score'] = (2. * ((self.training_stat['precision'] * self.training_stat['sensitivity']) /
                                                (self.training_stat['precision'] + self.training_stat['sensitivity'])))
        assert(np.allclose([true_positives.sum() / (true_positives.sum() + false_positives.sum())],
                           [self.training_stat['accuracy']]))

    def _write_training_stat_to_log(self):
        with open(self.log_file_path, 'a') as f:
            print((f"{self.training_unique_id},"
                   f"{self.training_stat['wall_time']:.01f},"
                   f"{self.training_stat['cpu_time']:.01f},"
                   f"{self.training_stat['epochs']},"
                   f"{self.training_stat['accuracy']:.04f},"
                   f"{np.amin(self.training_stat['precision']):.04f},"
                   f"{np.amin(self.training_stat['sensitivity']):.04f},"
                   f"{np.amin(self.training_stat['specificity']):.04f},"
                   f"{np.amin(self.training_stat['f1_score']):.04f},"), file=f)

        # Write confusion matrix to its own file.
        sign_names = pd.read_csv(os.path.join(self.project_file_path, "signnames.csv"))
        df = pd.DataFrame(self.training_stat['confusion_matrix'])
        df.columns = sign_names['SignName']
        df['TP + FN'] = df.sum(axis=1)
        df.loc[df.index[-1] + 1] = df.sum(axis=0)
        df.insert(0, column='', value=np.concatenate((sign_names['SignName'].values, ['TP + FP'])))
        df.to_csv(os.path.join(self.training_file_path, "confusion_matrix.csv"), index=False)

        # Write detailed training statistics to its own file
        detailed_stat = dict((k, self.training_stat[k]) for k in ('precision', 'sensitivity',
                                                                  'specificity', 'f1_score'))
        df = pd.DataFrame.from_dict(detailed_stat, orient='index')
        df.columns = '' + sign_names['SignName']
        df.to_csv(os.path.join(self.training_file_path, "detailed_stat.csv"))

    def _create_training_file_path(self):
        from datetime import datetime
        self.training_unique_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.training_file_path = os.path.join(self.output_file_path, self.training_unique_id)
        assert(os.path.exists(self.training_file_path) is False)
        os.makedirs(self.training_file_path)

    def _write_graph_to_file(self, nn_graph):
        with open(os.path.join(self.training_file_path, "NeuralNetworkOperations.txt"), 'w') as f:
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


if __name__ == '__main__':
    # Example: Run the TrafficSignTrainer with variants of the LeNet graph.
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
