from argparse import ArgumentParser

from sav_benchmark import benchmark

"""
The structure of the {GT_ROOT} can be either of the follow two structures. 
{GT_ROOT} and {PRED_ROOT} should be of the same format

1. SA-V val/test structure
    {GT_ROOT}  # gt root folder
        ├── {video_id}
        │     ├── 0000               # all masks associated with obj 0000
        │     │    ├── {frame_id}.png    # mask for object 000 in {frame_id} (binary mask)
        │     │    └── ...
        │     ├── 0001               # all masks associated with obj 0001
        │     ├── 0002               # all masks associated with obj 0002
        │     └── ...
        ├── {video_id}
        ├── {video_id}
        └── ...

2. Similar to DAVIS structure:

    {GT_ROOT}  # gt root folder
        ├── {video_id}
        │     ├── {frame_id}.png          # annotation in {frame_id} (may contain multiple objects)
        │     └── ...
        ├── {video_id}
        ├── {video_id}
        └── ...
"""


parser = ArgumentParser()
parser.add_argument(
    "--gt_root",
    required=True,
    help="Path to the GT folder. For SA-V, it's sav_val/Annotations_6fps or sav_test/Annotations_6fps",
)
parser.add_argument(
    "--pred_root",
    required=True,
    help="Path to a folder containing folders of masks to be evaluated, with exactly the same structure as gt_root",
)
parser.add_argument(
    "-n", "--num_processes", default=16, type=int, help="Number of concurrent processes"
)
parser.add_argument(
    "-s",
    "--strict",
    help="Make sure every video in the gt_root folder has a corresponding video in the prediction",
    action="store_true",
)
parser.add_argument(
    "-q",
    "--quiet",
    help="Quietly run evaluation without printing the information out",
    action="store_true",
)

# https://github.com/davisvideochallenge/davis2017-evaluation/blob/d34fdef71ce3cb24c1a167d860b707e575b3034c/davis2017/evaluation.py#L85
parser.add_argument(
    "--do_not_skip_first_and_last_frame",
    help="In SA-V val and test, we skip the first and the last annotated frames in evaluation. "
    "Set this to true for evaluation on settings that doen't skip first and last frames",
    action="store_true",
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Evaluating {args.pred_root}")
    benchmark(
        [args.gt_root],
        [args.pred_root],
        args.strict,
        args.num_processes,
        verbose=not args.quiet,
        skip_first_and_last=not args.do_not_skip_first_and_last_frame,
    )
