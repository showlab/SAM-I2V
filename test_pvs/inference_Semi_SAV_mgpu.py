import cv2
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from scipy.ndimage import label
from utils import *
from natsort import natsorted
import os

dataset_image_path = '/workspace/i2v/data/sav_test/JPEGImages_24fps'
dataset_anno_path = '/workspace/i2v/data/sav_test/Annotations_6fps'
dataset_gt_path = '/workspace/i2v/data/sav_test/mask_info/'

dataset_dir = sorted(os.listdir(dataset_image_path))


def _save_mask(path_, mask_np_):
    cv2.imwrite(path_, mask_np_)


def process_sequences(args_settings, gpu_id, seq_list):

    save_pool = ThreadPoolExecutor(max_workers=8)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    torch.cuda.set_device(0)

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    from i2v.build_i2v import build_i2v_video_predictor

    input = args_settings.input
    ckpt = args_settings.ckpt
    yaml = args_settings.yaml
    save_dir_name = args_settings.save_dir_name

    model_cfg = f"../i2v/configs/{yaml}"
    checkpoint = f"../checkpoints/{ckpt}.pt"
    checkpoint = os.path.expanduser(checkpoint)
    save_path = f"./output_semi/{save_dir_name}/Semi_SAVTest_{ckpt}_{input}/"

    predictor = build_i2v_video_predictor(model_cfg, checkpoint)

    for n_video, seq_name in enumerate(tqdm(seq_list, desc=f"GPU {gpu_id}: ")):

        seq_path = os.path.join(dataset_image_path, seq_name)
        seq_dir = natsorted(glob(seq_path + '/*'))

        frame_name = natsorted(os.listdir(seq_path))
        frame_name = [ i.split('.')[0] for i in frame_name]

        gt_mask_path = os.path.join(dataset_gt_path, seq_name)
        instance_list = natsorted(os.listdir(gt_mask_path))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for save_id in instance_list:
                save_id = save_id.split('.')[0]

                instance_save_path = os.path.join(save_path,seq_name,save_id)
                if os.path.exists(instance_save_path) and len(os.listdir(instance_save_path)) != 0:
                    continue
                else:
                    os.makedirs(instance_save_path, exist_ok=True)

                state = predictor.init_state(
                    video_path=seq_path,
                    async_loading_frames=False,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                )
                anno_point_record = []
                gt_mask = np.load(os.path.join(gt_mask_path, save_id + '.npy'), allow_pickle=True).item()
                gt_mask = check_mask(gt_mask)

                start_idx = 0
                for k, v in gt_mask.items():
                    if v[0].any():
                        start_idx = k
                        break

                for idx in range(len(seq_dir)):
                    if idx not in gt_mask.keys():
                        gt_mask[idx] = {}
                        gt_mask[idx][0] = np.zeros_like(gt_mask[start_idx][0])

                # (a) 3-click prompt
                if input == "3c":

                    num_click = 3

                    interact_points, gt_state = select_interact_point_center2(gt_mask[start_idx][0])

                    anno_point_record.append(interact_points['0'])

                    prompt_iou = []
                    point = interact_points['0']
                    video_segments = {}

                    input_points = torch.tensor(point.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    input_labels = torch.tensor([[1]], dtype=torch.int32)

                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                                   frame_idx=start_idx, obj_id=int(0),
                                                                                   points=input_points,
                                                                                   labels=input_labels)

                    input_points = torch.tensor(point.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    input_labels = torch.tensor([[1]], dtype=torch.int32)

                    video_segments[0] = {
                        out_obj_id: (masks[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(object_ids)
                    }

                    frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])

                    prompt_iou.append(frame_iou.item())

                    for i in range(num_click - 1):
                        point, label = get_next_point(torch.tensor(gt_mask[start_idx][0]).unsqueeze(0).unsqueeze(0),
                                                      torch.tensor(video_segments[0][0]).unsqueeze(0), 'center')

                        anno_point_record.append(np.array(point[0][0]))

                        input_points = torch.cat((input_points, point), dim=1)
                        input_labels = torch.cat((input_labels, label), dim=1)

                        frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                                       frame_idx=start_idx, obj_id=0,
                                                                                       points=input_points,
                                                                                       labels=input_labels)
                        video_segments = {}
                        video_segments[0] = {out_obj_id: (masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(object_ids)}

                        frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])
                        prompt_iou.append(frame_iou.item())

                    max_idx = np.argmax(prompt_iou).item()
                    input_points = input_points[:, :max_idx + 1, :]
                    input_labels = input_labels[:, :max_idx + 1]
                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                                   frame_idx=start_idx, obj_id=0,
                                                                                   points=input_points,
                                                                                   labels=input_labels)

                # (b) bounding-box prompt
                elif input == "bb":
                    binary_image = np.uint8(gt_mask[start_idx][0]) * 255
                    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    all_points = np.concatenate([contour for contour in contours])

                    x, y, w, h = cv2.boundingRect(all_points)
                    box = np.array([x, y, x + w, y + h], dtype=np.float32)

                    frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,
                                                                                   frame_idx=start_idx, obj_id=0,
                                                                                   box=box)
                    video_segments = {}
                    video_segments[0] = {out_obj_id: (masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(object_ids)}

                    frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])

                # (c) ground-truth-mask prompt
                elif input == "gm":
                    frame_idx, object_ids, masks = predictor.add_new_mask(inference_state=state, frame_idx=start_idx,
                                                                          obj_id=0, mask=gt_mask[start_idx][0])

                    video_segments = {}
                    video_segments[0] = {out_obj_id: (masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(object_ids)}

                    frame_iou = cal_maskIoU(video_segments[0][0], gt_mask[start_idx][0])

                # (d) other prompt
                else:
                    raise NotImplementedError

                # propagate in video
                gpu_masks = []
                frame_indices = []
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0)
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    gpu_masks.append(video_segments[out_frame_idx][0][0])  # CUDA Tensor
                    frame_indices.append(out_frame_idx)

                stacked_np = (torch.stack(gpu_masks, 0).cpu().numpy() * 255.).astype(np.uint8)

                for i, idx in enumerate(frame_indices):
                    save_path_png = os.path.join(instance_save_path, frame_name[idx] + '.png')
                    save_pool.submit(_save_mask, save_path_png, stacked_np[i])

    save_pool.shutdown(wait=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--yaml",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--save_dir_name",
        required=True,
        type=str,
    )
    args_settings = parser.parse_args()

    mp.set_start_method('spawn')

    gpu_num = 8

    gpu_sequences = [[] for _ in range(gpu_num)]
    for idx, seq_name in enumerate(dataset_dir):
        gpu_sequences[idx % gpu_num].append(seq_name)

    print(f"Start SAVTest {args_settings.ckpt}_{args_settings.input}...")

    processes = []
    for gpu_id in range(gpu_num):
        seq_list = gpu_sequences[gpu_id]
        if not seq_list:
            continue

        p = mp.Process(
            target=process_sequences,
            args=(args_settings, gpu_id, seq_list)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Finished SAVTest {args_settings.ckpt}_{args_settings.input}.\nSaved results in {args_settings.save_dir_name}.")
