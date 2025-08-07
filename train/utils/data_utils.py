from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch. O 是大写的o，而不是0
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData

    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        From [TxBxCxHxW] to [(B*T)xCxHxW]
        """
        return self.img_batch.transpose(0, 1).flatten(0, 1)

    def flat_vid_batch(self, t_window) -> torch.FloatTensor:
        """
        [T x B x C x H x W] --> [(B*T) x C x t_window x H x W]
        """

        T, B, C, H, W = self.img_batch.shape

        sequences = []

        for t in range(T):
            start = t - t_window + 1
            end = t + 1

            if start >= 0:
                frames = self.img_batch[start:end]  # [t_window x B x C x H x W]
            else:
                pad_len = -start
                pad_frames = self.img_batch[t].unsqueeze(0).repeat(pad_len, 1, 1, 1, 1)  # [pad_len x B x C x H x W]
                frames = torch.cat([self.img_batch[0:end], pad_frames], dim=0)  # [t_window x B x C x H x W]

            frames = frames.permute(1, 0, 2, 3, 4)  # [B x t_window x C x H x W]

            sequences.append(frames)

        sequences = torch.stack(sequences, dim=1)  # [B x T x t_window x C x H x W]

        sequences = sequences.view(B * T, t_window, C, H, W)  # [(B*T) x t_window x C x H x W]

        sequences = sequences.permute(0, 2, 1, 3, 4)  # [(B*T) x C x t_window x H x W]

        return sequences

    def flat_vid_batch_with_interval(self, t_window, t_interval) -> torch.FloatTensor:
        """
        frame interval is: t_interval
        [T x B x C x H x W] --> [(B*T) x C x t_window x H x W]
        """

        T, B, C, H, W = self.img_batch.shape

        sequences = []

        for t in range(T):
            frames = []

            for i in range(t_window):
                idx = t - (t_window - 1 - i) * t_interval
                if idx >= 0:
                    frames.append(self.img_batch[idx])  # [B x C x H x W]
                else:
                    break

            while len(frames) < t_window:
                frames.append(self.img_batch[t])  # [B x C x H x W]

            frames = torch.stack(frames, dim=0)  # [t_window x B x C x H x W]

            frames = frames.permute(1, 0, 2, 3, 4)  # [B x t_window x C x H x W]

            sequences.append(frames)

        sequences = torch.stack(sequences, dim=1)  # [B x T x t_window x C x H x W]

        sequences = sequences.view(B * T, t_window, C, H, W)  # [(B*T) x t_window x C x H x W]

        sequences = sequences.permute(0, 2, 1, 3, 4)  # [(B*T) x C x t_window x H x W]

        return sequences


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
