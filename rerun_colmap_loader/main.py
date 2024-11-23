#!/usr/bin/env python3
"""Example of using Rerun to log and visualize the output of COLMAP's sparse reconstruction."""

from __future__ import annotations

import io
import os
import re
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import requests
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
from tqdm import tqdm

from .read_write_model import Camera, read_model  # type: ignore[attr-defined]

DATASET_DIR: Final = Path(__file__).parent.parent / "dataset"
DATASET_URL_BASE: Final = "https://storage.googleapis.com/rerun-example-datasets/colmap"
# When dataset filtering is turned on, drop views with less than this many valid points.
FILTER_MIN_VISIBLE: Final = 500


def scale_camera(
    camera: Camera, resize: tuple[int, int]
) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    # assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(
        camera.params[:2] * scale_factor, camera.params[2:] * scale_factor
    )

    return (
        Camera(camera.id, camera.model, new_width, new_height, new_params),
        scale_factor,
    )


def get_downloaded_dataset_path(dataset_name: str) -> Path:
    dataset_url = f"{DATASET_URL_BASE}/{dataset_name}.zip"

    recording_dir = DATASET_DIR / dataset_name
    if recording_dir.exists():
        return recording_dir

    os.makedirs(DATASET_DIR, exist_ok=True)

    zip_file = download_with_progress(dataset_url)

    with zipfile.ZipFile(zip_file) as zip_ref:
        progress = tqdm(
            zip_ref.infolist(),
            "Extracting dataset",
            total=len(zip_ref.infolist()),
            unit="files",
        )
        for file in progress:
            zip_ref.extract(file, DATASET_DIR)
            progress.update()

    return recording_dir


def download_with_progress(url: str) -> io.BytesIO:
    """Download file with tqdm progress bar."""
    chunk_size = 1024 * 1024
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))
    with tqdm(
        desc="Downloading dataset",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        zip_file = io.BytesIO()
        for data in resp.iter_content(chunk_size):
            zip_file.write(data)
            progress.update(len(data))

    zip_file.seek(0)
    return zip_file


def read_and_log_sparse_reconstruction(
    dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None
) -> None:
    print("Reading sparse COLMAP reconstruction")
    try:
        cameras, images, points3D = read_model(dataset_path / "sparse", ext=".bin")
    except FileNotFoundError:
        cameras, images, points3D = read_model(
            dataset_path / "sparse" / "0", ext=".bin"
        )

    print("Building visualization by logging to Rerun")

    if filter_output:
        # Filter out noisy points
        points3D = {
            id: point
            for id, point in points3D.items()
            if point.rgb.any() and len(point.image_ids) > 4
        }

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]), static=True)

    # Iterate through images (video frames) logging data related to each frame.
    for image in sorted(images.values(), key=lambda im: im.name):  # type: ignore[no-any-return]
        image_file = dataset_path.parent / "images" / image.name

        if not os.path.exists(image_file):
            continue

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        visible = [
            id != -1 and points3D.get(id) is not None for id in image.point3D_ids
        ]
        visible_ids = image.point3D_ids[visible]

        if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
            continue

        visible_xyzs = [points3D[id] for id in visible_ids]
        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        rr.set_time_sequence("frame", frame_idx)

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        rr.log(
            "points",
            rr.Points3D(points, colors=point_colors),
            # rr.AnyValues(error=point_errors),
        )

        # COLMAP's camera transform is "camera from world"
        rr.log(
            "camera",
            rr.Transform3D(
                translation=image.tvec,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
                from_parent=True,
                axis_length=0.5,
            ),
        )
        rr.log(
            "camera", rr.ViewCoordinates.RDF, static=True
        )  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        # assert camera.model == "PINHOLE"
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=camera.params[:2],
                principal_point=camera.params[2:],
                image_plane_distance=0.5,
            ),
        )

        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rr.log(
                "camera/image",
                rr.Image(bgr, color_model="BGR").compress(jpeg_quality=75),
            )
        else:
            rr.log(
                "camera/image",
                rr.EncodedImage(path=dataset_path.parent / "images" / image.name),
            )

        rr.log(
            "camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167])
        )


# The Rerun Viewer will always pass these two pieces of information:
# 1. The path to be loaded, as a positional arg.
# 2. A shared recording ID, via the `--recording-id` flag.
#
# It is up to you whether you make use of that shared recording ID or not.
# If you use it, the data will end up in the same recording as all other plugins interested in
# that file, otherwise you can just create a dedicated recording for it. Or both.
parser = ArgumentParser(
    description="""
Colmap reconstruction loader for Rerun.

To try it out, copy it in your $PATH as `rerun-colmap-loader`,
then open a colmap reconstruction with Rerun (`rerun `).
    """
)
parser.add_argument("filepath", type=str)
parser.add_argument("--recording-id", type=str)
args = parser.parse_args()


def main() -> None:
    is_file: bool = os.path.isfile(args.filepath)
    is_colmap_bin_file: bool = ".bin" in args.filepath

    # iterate through all files in the directory

    # Inform the Rerun Viewer that we do not support that kind of file.
    if not is_file or not is_colmap_bin_file:
        exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    rr.init(
        "rerun_example_external_data_loader_tfrecord", recording_id=args.recording_id
    )
    # The most important part of this: log to standard output so the Rerun Viewer can ingest it!
    rr.stdout()
    print("Loading COLMAP reconstruction")
    rr.log("text", rr.TextDocument("Loading COLMAP reconstruction"))
    # read_and_log_sparse_reconstruction(args.filepath, filter_output=False, resize=None)


if __name__ == "__main__":
    main()
