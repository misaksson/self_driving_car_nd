import os
import cv2
from shutil import copyfile

dir_path = "../output/hard_negative_mining/"
tp_path = os.path.join(dir_path, "TP")
fp_path = os.path.join(dir_path, "FP")

all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

idx = 0
while idx < len(all_files):
    file = os.path.join(dir_path, all_files[idx])
    is_tp = os.path.isfile(os.path.join(tp_path, all_files[idx]))
    is_fp = os.path.isfile(os.path.join(fp_path, all_files[idx]))

    patch = cv2.imread(file)
    patch = cv2.resize(patch, (500, 500))
    if is_tp:
        cv2.rectangle(patch, (0, 0), (patch.shape[1], patch.shape[0]), (0, 255, 0), 3)
    if is_fp:
        cv2.line(patch, (0, 0), (patch.shape[1], patch.shape[0]), (0, 0, 255), 3)
        cv2.line(patch, (0, patch.shape[0]), (patch.shape[1], 0), (0, 0, 255), 3)
    cv2.imshow("Patch", patch)
    k = cv2.waitKey() & 0xff

    if k == 27:
        break  # Esc
    elif k == 81:
        # Left arrow, move towards FP
        if is_tp:
            os.remove(os.path.join(tp_path, all_files[idx]))
        else:
            copyfile(file, os.path.join(fp_path, all_files[idx]))
    elif k == 83:
        # Right arrow, move towards FP
        if is_fp:
            os.remove(os.path.join(fp_path, all_files[idx]))
        else:
            copyfile(file, os.path.join(tp_path, all_files[idx]))
    elif k == 82:
        # Up arrow, step back
        idx -= 1
    else:
        idx += 1
