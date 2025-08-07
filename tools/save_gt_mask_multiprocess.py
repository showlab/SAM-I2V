from glob import glob
import cv2
import numpy as np
import os
from multiprocessing import Pool
from natsort import natsorted

def matrix_in_dict_values(dictionary, matrix):
    matrix_np = np.array(matrix)
    for value in dictionary.values():
        if np.array_equal(value, matrix_np):
            return True
    return False

def find_keys_for_value(dictionary, target_value):
    keys = [key for key, value in dictionary.items() if np.array_equal(value, target_value)]
    return keys

def load_mask(mask, color_dict):
    instance_idx = len(color_dict)
    pixels = mask.reshape(-1, mask.shape[-1])
    unique_colors = np.unique(pixels, axis=0)
    gt_mask = {}
    for color in unique_colors:
        if color.sum() != 0:
            if not matrix_in_dict_values(color_dict, color):
                color_dict[instance_idx] = color
                cur_mask = np.all(mask == color, axis=-1)
                gt_mask[instance_idx] = cur_mask
                instance_idx += 1
            else:
                key = find_keys_for_value(color_dict, color)[0]
                cur_mask = np.all(mask == color, axis=-1)
                gt_mask[key] = cur_mask
    return gt_mask

def save_gt_mask(path):
    print('Processing sequence: ' + path.split('/')[-1])

    save_root = '/workspace/i2v/data/sav_test/mask_info'
    save_path = os.path.join(save_root, os.path.basename(path))
    os.makedirs(save_path, exist_ok=True)

    obj_dirs = natsorted([d for d in glob(os.path.join(path, '*')) if os.path.isdir(d)])
    gt_state = {}
    instance_color = {}

    for obj_id, obj_dir in enumerate(obj_dirs):
        pngs = natsorted(glob(os.path.join(obj_dir, '*.png')))
        for png_path in pngs:
            frame_name = os.path.splitext(os.path.basename(png_path))[0]
            frame_idx = int(frame_name)
            mask = cv2.imread(png_path)

            if mask is None:
                continue

            if frame_idx not in gt_state:
                gt_state[frame_idx] = {}

            color = np.array([obj_id + 1, 0, 0], dtype=np.uint8)
            binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
            colored_mask = np.zeros_like(mask)
            colored_mask[binary_mask] = color

            instance_mask = load_mask(colored_mask, instance_color)
            for k, v in instance_mask.items():
                gt_state[frame_idx][k] = v

    for i in gt_state.keys():
        if gt_state[i]:
            H, W = list(gt_state[i].values())[0].shape
            break

    for instance_id in range(len(instance_color)):
        cur_mask = {}
        for i in gt_state.keys():
            if instance_id in gt_state[i]:
                cur_mask[i] = {0: gt_state[i][instance_id]}
            else:
                cur_mask[i] = {0: np.zeros((H, W), dtype=bool)}
        np.save(os.path.join(save_path, f'{instance_id:03d}.npy'), cur_mask)

def main():
    dataset_path = '/workspace/i2v/data/sav_test/Annotations_6fps'
    dataset_dir = sorted(os.listdir(dataset_path))
    dataset_dir = [os.path.join(dataset_path, i) for i in dataset_dir if os.path.isdir(os.path.join(dataset_path, i))]

    with Pool(128) as p:
        p.map(save_gt_mask, dataset_dir)

    print("All tasks processed.")

if __name__ == '__main__':
    main()
