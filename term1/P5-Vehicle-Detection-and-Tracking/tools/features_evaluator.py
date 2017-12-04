"""Feature evaluation tool

Extracts and run the classifier over all permutations of specified feature
extraction arguments (exhaustive search).
"""
import sys
import itertools

sys.path.append("../src/")
from trainer import Trainer


# Dictionary with lists of variants to try for each argument.

# Run 1: Searching for good HOG parameters
# variants = {'color_space': ['HSV'],
#             'spatial_size': [(32, 32)],
#             'hist_bins': [32],
#             'orient': range(6, 13, 3),
#             'pix_per_cell': range(4, 13, 4),
#             'cell_per_block': range(1, 4, 1),
#             'hog_channels': [[0, 1, 2]],
#             'spatial_feat': [False],
#             'hist_feat': [False],
#             'hog_feat': [True],
#             }
# Best result: 'orient': 12, 'pix_per_cell': 8, 'cell_per_block': 3, ---> accuracy 0.98818


# Run 2: Checking if spatial and/or histogram features add significant information
# variants = {'color_space': ['HSV'],
#             'spatial_size': [(32, 32)],
#             'hist_bins': [32],
#             'orient': [12],
#             'pix_per_cell': [8],
#             'cell_per_block': [3],
#             'hog_channels': [[0, 1, 2]],
#             'spatial_feat': [False, True],
#             'hist_feat': [False, True],
#             'hog_feat': [True],
#             }
# Best result: 'spatial_feat': True, 'hist_feat': True,  ---> 99381

# Run 3: Checking if smaller or larger spatial size give better result.
# variants = {'color_space': ['HSV'],
#             'spatial_size': [(8, 8), (16, 16), (32, 32), (48, 48), (64, 64)],
#             'hist_bins': [32],
#             'orient': [12],
#             'pix_per_cell': [8],
#             'cell_per_block': [3],
#             'hog_channels': [[0, 1, 2]],
#             'spatial_feat': [True],
#             'hist_feat': [True],
#             'hog_feat': [True],
#             }
# Best result: 'spatial_size': (16, 16) ---> 0.99325
# This is actually worse than before, so spatial features doesn't seem to add any significant value.

# Run 4: Try different color spaces
#variants = {'color_space': ['HSV', 'BGR', 'LUV', 'HLS', 'YUV', 'YCrCb'],
#            'spatial_size': [(32, 32)],
#            'hist_bins': [32],
#            'orient': [12],
#            'pix_per_cell': [8],
#            'cell_per_block': [3],
#            'hog_channels': [[0, 1, 2]],
#            'spatial_feat': [False],
#            'hist_feat': [True],
#            'hog_feat': [True],
#            }
# Best result: 'color_space': 'YUV' --->  0.99381

# Run 5: Check if all HOG channels are needed
# variants = {'color_space': ['YUV'],
#             'spatial_size': [(32, 32)],
#             'hist_bins': [32],
#             'orient': [12],
#             'pix_per_cell': [8],
#             'cell_per_block': [3],
#             'hog_channels': [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]],
#             'spatial_feat': [False],
#             'hist_feat': [True],
#             'hog_feat': [True],
#             }
# Best result: 'hog_channels': [0, 1, 2] ---> 0.99353
# The result is however roughly the same without the U-channel so I might consider dropping it.

# Run 6: Try different size of color histograms
#variants = {'color_space': ['YUV'],
#            'spatial_size': [(32, 32)],
#            'hist_bins': [8, 16, 32, 64, 128],
#            'orient': [12],
#            'pix_per_cell': [8],
#            'cell_per_block': [3],
#            'hog_channels': [[0, 1, 2]],
#            'spatial_feat': [False],
#            'hist_feat': [True],
#            'hog_feat': [True],
#            }
# Best result: 'hist_bins': >32 ---> 0.99325
# Still looks like the color histogram add some non-negligible performance improvement.


# Run 7: Fine-tune HOG parameters.
# variants = {'color_space': ['YUV'],
#             'spatial_size': [(32, 32)],
#             'hist_bins': [32],
#             'orient': [11, 12, 13],
#             'pix_per_cell': [7, 8, 9],
#             'cell_per_block': [2, 3, 4, 5],
#             'hog_channels': [[0, 1, 2]],
#             'spatial_feat': [False],
#             'hist_feat': [True],
#             'hog_feat': [True],
#             }
# Best result:  'hist_bins': 32, 'orient': 12, 'pix_per_cell': 8, 'cell_per_block': 2, 'hog_channels': [0, 1, 2], ---> 0.99494


variants = {'color_space': ['YUV'],
            'spatial_size': [(32, 32)],
            'hist_bins': [32],
            'orient': [12],
            'pix_per_cell': [8],
            'cell_per_block': [2],
            'hog_channels': [[0, 1, 2]],
            'spatial_feat': [False],
            'hist_feat': [True],
            'hog_feat': [True],
            }


n_variants = 1
arg_name_list = []
variants_list = []
for arg_name, value in variants.items():
    n_variants *= len(value)
    arg_name_list.append(arg_name)
    variants_list.append(value)

print(f"Number of variants to run {n_variants}")

trainer = Trainer()
with open("../output/feature_evaluator.txt", "a") as fid:
    for idx, values in enumerate(itertools.product(*variants_list)):
        print(f"Variant {idx + 1} / {n_variants}")
        feature_extractor_args = dict()
        for idx, arg_name in enumerate(arg_name_list):
            feature_extractor_args[arg_name] = values[idx]
        trainer.extract_features(feature_extractor_args)
        _, _, _, accuracy_score = trainer.train(C=2.0)
        print(f"{accuracy_score:.5f} {feature_extractor_args}", file=fid, flush=True)
    print("---", file=fid, flush=True)  # Separating runs in the log
