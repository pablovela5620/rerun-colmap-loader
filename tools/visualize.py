import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
from rerun_loader_colmap.main import read_and_log_sparse_reconstruction


def check_colmap_files_exist(fp: Path) -> bool:
    required_files: list[str] = ["images", "cameras", "points3D"]
    # makes sure that the images, cameras, and points3D files are present
    all_files: list[Path] = list(fp.parent.glob(f"*{fp.suffix}"))
    all_stems: list[str] = [f.stem for f in all_files]
    missing_files: list[str] = [f for f in required_files if f not in all_stems]
    return len(missing_files) == 0


def main() -> None:
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
    parser.add_argument("filepath", type=Path)
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "visualize")

    print("Loading COLMAP reconstruction")
    files_exist = check_colmap_files_exist(args.filepath)
    if not files_exist:
        exit(rr.EXTERNAL_DATA_LOADER_INCOMPATIBLE_EXIT_CODE)

    read_and_log_sparse_reconstruction(
        args.filepath.parent,
        filter_output=False,
        resize=None,
        extention=args.filepath.suffix,
    )
    rr.script_teardown(args=args)


if __name__ == "__main__":
    main()
