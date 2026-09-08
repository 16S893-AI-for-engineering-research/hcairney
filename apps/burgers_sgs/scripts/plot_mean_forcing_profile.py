#!/usr/bin/env python3
"""Plot mean applied action with temporal block standard-error shading.

Use action start times, exclude times at or before the warmup cutoff, and
average snapshots within complete [left, right) blocks containing at least
two decisions. The plotted mean averages those block means; the band is their
sample standard deviation divided by sqrt(block count). Discard partial blocks.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def read_json(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return document


def read_actions(path):
    """Read one episode with a fixed grid and increasing decision times."""
    import numpy as np

    frames = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"decision_index", "start_time", "end_time", "x", "applied_action"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            try:
                step = int(row["decision_index"])
                start, end, x, action = (
                    float(row[key]) for key in
                    ("start_time", "end_time", "x", "applied_action")
                )
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid numeric value on line {line}") from error
            if step < 0 or not all(map(math.isfinite, (start, end, x, action))) or end <= start:
                raise ValueError(f"invalid decision interval or value on line {line}")
            if not frames or step != frames[-1][0]:
                if frames and (step <= frames[-1][0] or start < frames[-1][2]):
                    raise ValueError("decisions must increase with non-overlapping intervals")
                frames.append((step, start, end, [], []))
            frame = frames[-1]
            if (start, end) != frame[1:3]:
                raise ValueError(f"inconsistent decision times on line {line}")
            frame[3].append(x)
            frame[4].append(action)
    if not frames:
        raise ValueError("RL field history contains no data rows")
    grid = frames[0][3]
    if len(set(grid)) != len(grid) or any(frame[3] != grid for frame in frames):
        raise ValueError("spatial grid must be fixed with unique cell coordinates")
    return (np.asarray(grid), np.asarray([frame[1] for frame in frames]),
            np.asarray([frame[4] for frame in frames]), frames[-1][2])


def read_mean_forcing_profile(path, block_duration=None, statistics_start_time=None):
    """Return coordinates, mean, SE, and retained block count."""
    import numpy as np

    path = Path(path)
    summary = read_json(path.parent / "analysis_summary.json")
    if block_duration is None:
        block_duration = summary.get("block_duration", 10.0)
    try:
        block_duration = float(block_duration)
    except (TypeError, ValueError) as error:
        raise ValueError("block duration must be finite and positive") from error
    if not math.isfinite(block_duration) or block_duration <= 0:
        raise ValueError("block duration must be finite and positive")

    coordinates, times, actions, final_time = read_actions(path)
    window = summary.get("window", {})
    if statistics_start_time is None:
        statistics_start_time = window.get("start")
        if statistics_start_time is None:
            metadata = read_json(path.parent / "burgers_run_metadata.json")
            statistics_start_time = metadata.get("configuration", {}).get(
                "output", {}).get("statistics_start_time")
        if statistics_start_time is None:
            raise ValueError("warmup cutoff unavailable; supply --statistics-start-time")
    try:
        start = float(statistics_start_time)
        end = min(float(window.get("end", final_time)), final_time)
    except (TypeError, ValueError) as error:
        raise ValueError("analysis window must have finite END > START") from error
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("analysis window must have finite END > START")
    blocks = []
    for block in range(int(math.floor((end - start) / block_duration + 1e-12))):
        left = start + block * block_duration
        right = left + block_duration
        mask = (times > start) & (times >= left) & (times < right)
        if np.count_nonzero(mask) >= 2:
            blocks.append(np.mean(actions[mask], axis=0))
    if len(blocks) < 2:
        raise ValueError("at least two complete blocks containing two decisions each are required")
    blocks = np.asarray(blocks)
    return coordinates, blocks.mean(axis=0), blocks.std(axis=0, ddof=1) / math.sqrt(len(blocks)), len(blocks)


def make_figure(path, block_duration=None, statistics_start_time=None, xlim=None, ylim=None):
    import matplotlib.pyplot as plt

    x, mean, standard_error, _ = read_mean_forcing_profile(
        path, block_duration, statistics_start_time
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(x, mean, linewidth=1.5, label="Mean applied action")
    axis.fill_between(x, mean - standard_error, mean + standard_error,
                      alpha=0.25, label=r"Temporal block standard error ($\pm 1$ SE)")
    axis.set_title("Temporal mean forcing profile")
    axis.set_xlabel(r"$x$")
    axis.set_ylabel("Mean applied action")
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fields_file", type=Path, help="burgers_rl_fields.csv for one episode")
    parser.add_argument("--block-duration", type=float,
                        help="block duration: CLI, then analysis_summary.json, then 10")
    parser.add_argument("--statistics-start-time", type=float,
                        help="warmup cutoff: CLI, then summary window.start, then run metadata")
    parser.add_argument("-o", "--output", type=Path, default=Path("mean_forcing_profile.png"),
                        help="output filename beside the CSV (default: mean_forcing_profile.png)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"))
    parser.add_argument("--show", action="store_true")
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        figure = make_figure(arguments.fields_file, arguments.block_duration,
                             arguments.statistics_start_time, arguments.xlim, arguments.ylim)
        figure.savefig(arguments.fields_file.parent / arguments.output.name, dpi=arguments.dpi)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        parser.error(str(error))
    if arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
