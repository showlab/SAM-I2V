import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import label
from PIL import Image
import os
import shutil
from tqdm import tqdm


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    mask1 = mask.convert("P", palette=Image.ADAPTIVE, colors=256)
    palette = mask1.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def load_masks_from_dir_other_datasets(
    input_mask_dir, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(input_mask_dir):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def mask2points(mask_path):
    mask = cv2.imread('001.png')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    points_list = cnt.squeeze().tolist()
    print(points_list)


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    elif method == "positive":
        return sample_positive_point_from_error(gt_masks, pred_masks)


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - gt_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool
    - pred_masks: [B, 1, H_im, W_im] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 2], dtype=torch.float, contains (x, y) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def select_interact_point_center2(mask):
    
    interact_points = {}
    gt_state = mask2ploy((mask.astype(np.uint8)*255.)[:,:,np.newaxis].repeat(3,axis=2))
    
    rows, cols = np.where(mask)

    mean_row = rows.mean()
    mean_col = cols.mean()

    distances = np.sqrt((rows - mean_row) ** 2 + (cols - mean_col) ** 2)
    min_index = np.argmin(distances)
    center_row = rows[min_index]
    center_col = cols[min_index]
    
    interact_points['0'] = np.array([center_col,center_row])

    return interact_points, gt_state


def sample_positive_point_from_error(gt_masks, pred_masks):
   
    non_overlapping = np.array(gt_masks[0][0]).astype(int) - np.array(pred_masks[0][0]).astype(int)

    '''COMMENTS: For visualization '''
    non_overlapping_vis = non_overlapping.copy()
    non_overlapping_vis[non_overlapping_vis== -1] = 0
    non_overlapping_vis[non_overlapping_vis== 0] = 125
    non_overlapping_vis[non_overlapping_vis== 1] = 255
    
    gt_masks_vis = np.array(gt_masks[0][0]).astype(int) * 255
    pred_masks_vis = np.array(pred_masks[0][0]).astype(int) * 255

    res_vis = np.concatenate((gt_masks_vis,pred_masks_vis,non_overlapping_vis),axis=1)
    cv2.imwrite('visualization_error.jpg',res_vis)

    pos_click = (non_overlapping == 1)
    
    labeled_array_pos, num_features_pos = label(pos_click)
    
    area = []
    for i in range(1, num_features_pos+1):
        area.append((labeled_array_pos == i).sum())

    area_sort = area.copy()
    area_sort.sort(reverse=True)

    max_area = area_sort[0]
    
    idx1 = np.where(area == max_area)[0]

    idx = idx1[0]
    mask1 = (labeled_array_pos == idx+1)
    
    rows, cols = np.where(mask1)

    mean_row = rows.mean()
    mean_col = cols.mean()

    distances = np.sqrt((rows - mean_row) ** 2 + (cols - mean_col) ** 2)
    min_index = np.argmin(distances)
    center_row = rows[min_index]
    center_col = cols[min_index]
    
    return torch.tensor([center_col,center_row]).unsqueeze(0).unsqueeze(0), torch.tensor([1]).unsqueeze(0)


def check_mask(mask):
    H,W = mask[0][0].shape
    for key in mask.keys():
        
        if len(mask[key]) == 0:
            mask[key] = {}
            mask[key][0] = np.zeros((H,W),dtype=bool)
        else:
            mask[key][0] = mask[key][0].astype(bool)
    return mask


''' metric '''


def cal_maskIoU(result, gt):
    eps = 1e-6
    intersection = np.logical_and(result, gt)
    union = np.logical_or(result, gt)
    iou = np.sum(intersection) / (np.sum(union) + eps)
    if np.isnan(iou):
        iou = 0
    return iou


def cal_F1score(y_pred, y_true):
    # Calculate True Positives, False Positives, and False Negatives
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    if np.isnan(f1_score):
        f1_score = 0
    return f1_score


''' for visualization''' 


def visualize_gt_and_result(image_path, pred, gt, save_name):
    im = cv2.imread(image_path)

    pred = np.repeat(pred[0][:,:,np.newaxis],3,2) * 255.
    gt = np.repeat(gt[:,:,np.newaxis],3,2) * 255.

    res = np.concatenate((im,gt,pred),axis=1)
    cv2.imwrite(save_name,res)


def visualize_points(image_path, points, state):

    image = cv2.imread(image_path)
    for point in points:
        cv2.circle(image, point.astype(int), 2, (0,0,255), 5)

    cv2.imwrite('visualization.jpg',image)


def visualize_mask_points(image_path, gt, res, points, state, frame_id):

    image = cv2.imread(image_path)
    gt_vis = (gt.astype(int)[:,:,np.newaxis] * 255.).astype(np.uint8).repeat(3,axis=2)
    res_vis = (res[0][0].astype(int)[:,:,np.newaxis] * 255.).astype(np.uint8).repeat(3,axis=2)

    for point in points:
        cv2.circle(image, point.astype(int), 2, (0,0,255), 5)
        cv2.circle(gt_vis, point.astype(int), 2, (0,0,255), 5)
    
    vis = np.concatenate((image,gt_vis,res_vis),axis=0)

    cv2.imwrite('./visualization/visualization_'+str(frame_id)+'.jpg',vis)


def show_mask(mask, image, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    return mask_image[:,:,:3]


def visualization_results(seq_dir, video_segments, vis_frame_stride = 1, alpha = 0.6):

    if os.path.exists('./results/'):
        shutil.rmtree('./results/')

    os.makedirs('./results/')

    for out_frame_idx in range(0, len(seq_dir), vis_frame_stride):
        image = np.array(Image.open(seq_dir[out_frame_idx]))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():

            res = (show_mask(out_mask, image, obj_id=out_obj_id) * 255.).astype(np.uint8)
            image = cv2.addWeighted(res, alpha, image, 1 - alpha, 0)

        plt.imsave(os.path.join('./results/', str(out_frame_idx).zfill(6)+'.jpg'),image)


''' other tools'''


def load_gt_mask(anno_dir):

    gt_state = {}
    for frame_idx, mask_path in tqdm(enumerate(anno_dir)):
        mask = cv2.imread(mask_path)
        gt_state[frame_idx] = load_mask(mask)

    return None


def load_mask(mask):

    pixels = mask.reshape(-1, mask.shape[-1])
    unique_colors = np.unique(pixels, axis=0)
    num_instance = len(unique_colors) - 1

    gt_mask = {}
    instance_id = 0
    for color in unique_colors:
        if color.sum() != 0:
            cur_mask = np.all(mask==color, axis = -1)
            gt_mask[instance_id] = cur_mask
            instance_id += 1
    
    return gt_mask


def mask2ploy(mask):

    pixels = mask.reshape(-1, mask.shape[-1])
    unique_colors = np.unique(pixels, axis=0)
    num_instance = len(unique_colors) - 1

    state = {}
    ploy_instance = {}
    mask_instance = {}
    color_instance = {}
    instance_id = 0

    for color in unique_colors:
        if color.sum() != 0:
            cur_mask = np.all(mask==color, axis = -1).astype(np.uint8)
            contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 所有联通区域
            ploy_instance[str(instance_id)] = contours
            mask_instance[str(instance_id)] = cur_mask
            color_instance[str(instance_id)] = color
            instance_id += 1

    state['mask'] = mask_instance
    state['ploy'] = ploy_instance
    state['color'] = color_instance
    return state
