import argparse
import re
from pathlib import Path

import pandas as pd
import torch


def get_global_step(checkpoint):
    """Extract global_step from a Lightning checkpoint."""
    return checkpoint.get("global_step", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing version_* directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="checkpoints.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    rows = []

    for version_dir in sorted(
        args.model_dir.glob("version_*"),
        key=lambda p: int(re.search(r"version_(\d+)", p.name).group(1)),
    ):
        checkpoints_dir = version_dir / "checkpoints"

        if not checkpoints_dir.is_dir():
            continue

        version = int(
            re.search(r"version_(\d+)", version_dir.name).group(1)
        )

        for checkpoint_path in sorted(checkpoints_dir.glob("*.ckpt")):
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                )

                global_step = get_global_step(checkpoint)

                rows.append(
                    {
                        "version": version,
                        "filename": checkpoint_path.name,
                        "global_step": global_step,
                    }
                )

            except Exception as exc:
                print(
                    f"Warning: could not read "
                    f"{checkpoint_path}: {exc}"
                )

    df = pd.DataFrame(rows)

    # Ordena pelo global_step
    df = df.sort_values(
        ["global_step", "version", "filename"],
        na_position="last",
    ).reset_index(drop=True)

    df.to_csv(args.output, index=False)

    print(df)
    print(f"\nCSV saved to: {args.output}")


if __name__ == "__main__":
    main()

