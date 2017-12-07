"""Classifier evaluation tool

Train the classifier and validate the result for all permutations of the
argument variants.
"""

import sys
import itertools
import time
import pickle
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../src/")
from trainer import Trainer


# Dictionary with lists of values to try for each argument.
variants = {'min_samples_split': [0.00237],
            'max_features': ['sqrt'],
            'n_estimators': [10],
            'criterion': ["gini"],
            'max_depth': [100],
            }

n_variants = 1
arg_name_list = []
variants_list = []
for arg_name, value in variants.items():
    n_variants *= len(value)
    arg_name_list.append(arg_name)
    variants_list.append(value)

print(f"Number of variants to run {n_variants}")

trainer = Trainer(file_sets=[0])
trainer.extract_features()

with open("../output/classifier_evaluator.txt", "a") as log_fid:
    for variant_idx, values in enumerate(itertools.product(*variants_list)):
        print(f"Variant {variant_idx + 1} / {n_variants}")
        classifier_args = dict()
        for idx, arg_name in enumerate(arg_name_list):
            classifier_args[arg_name] = values[idx]
        classifier = RandomForestClassifier(**classifier_args)
        classifier, feature_scaler, feature_extractor_args, accuracy_score = trainer.train(classifier)

        # Save classifier to file.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = f"../output/classifier{timestamp}.p"
        with open(file_path, "wb") as pickle_fid:
            print("Writing classifier to", file_path)
            pickle.dump((classifier, feature_scaler, feature_extractor_args), pickle_fid)

        print(f"{accuracy_score:.5f}, {classifier_args}, {file_path}", file=log_fid, flush=True)
    print("---", file=log_fid, flush=True)  # Separating runs in the log
