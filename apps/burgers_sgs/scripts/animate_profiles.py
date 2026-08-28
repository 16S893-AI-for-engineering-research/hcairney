#!/usr/bin/env python3
"""Animate Burgers velocity profiles written by run_solver."""

import argparse
import csv
import math
from pathlib import Path


def read_profiles(path):
    """Return ``(step, time, x, u)`` tuples for all frames in *path*."""
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"step", "time", "x", "cell_average"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required column(s): {missing}")

        frames = []
        current_step = None
        current_time = None
        cell_centers = []
        cell_averages = []

        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["step"])
                time = float(row["time"])
                cell_center = float(row["x"])
                cell_average = float(row["cell_average"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric value on line {line_number}"
                ) from error

            if step < 0:
                raise ValueError(f"negative step on line {line_number}")
            if not all(math.isfinite(value) for value in
                       (time, cell_center, cell_average)):
                raise ValueError(f"non-finite value on line {line_number}")

            if current_step is None:
                current_step = step
                current_time = time
            elif step != current_step or time != current_time:
                frames.append(
                    (current_step, current_time, cell_centers, cell_averages)
                )
                current_step = step
                current_time = time
                cell_centers = []
                cell_averages = []

            cell_centers.append(cell_center)
            cell_averages.append(cell_average)

        if current_step is not None:
            frames.append(
                (current_step, current_time, cell_centers, cell_averages)
            )

    if not frames:
        raise ValueError("profile history contains no data rows")

    reference_grid = frames[0][2]
    if not reference_grid:
        raise ValueError("profile history contains an empty frame")
    previous_step = None
    previous_time = None
    for step, time, grid, _ in frames:
        if grid != reference_grid:
            raise ValueError(f"spatial grid changes at step {step}")
        if previous_step is not None and step <= previous_step:
            raise ValueError("profile steps must be strictly increasing")
        if previous_time is not None and time < previous_time:
            raise ValueError("profile times must be nondecreasing")
        previous_step = step
        previous_time = time

    return frames


def make_animation(path, interval_ms):
    """Create a Matplotlib figure and animation from a profile-history CSV."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required to animate the profiles"
        ) from error

    frames = read_profiles(path)
    all_values = [value for _, _, _, values in frames for value in values]
    minimum_u = min(all_values)
    maximum_u = max(all_values)
    u_span = maximum_u - minimum_u
    u_padding = 0.05 * u_span if u_span > 0.0 else 0.05 * max(
        1.0, abs(minimum_u)
    )

    cell_centers = frames[0][2]
    minimum_x = min(cell_centers)
    maximum_x = max(cell_centers)
    x_span = maximum_x - minimum_x
    x_padding = 0.02 * x_span if x_span > 0.0 else 0.5

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    line, = axes.plot([], [], linewidth=1.5)
    axes.set_xlim(minimum_x - x_padding, maximum_x + x_padding)
    axes.set_ylim(minimum_u - u_padding, maximum_u + u_padding)
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"Cell-average velocity $u$")
    axes.grid(alpha=0.3)
    title = axes.set_title("")

    def update(frame_index):
        step, time, grid, values = frames[frame_index]
        line.set_data(grid, values)
        title.set_text(f"Step {step}, t = {time:g}")
        return line, title

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    figure.tight_layout()
    return figure, animation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Animate a burgers_profiles.csv file produced by run_solver."
    )
    parser.add_argument("csv_file", type=Path, help="profile-history CSV file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="save as a .gif or .mp4 instead of only displaying the animation",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="saved-animation frame rate (default: 20)",
    )
    parser.add_argument(
        "--interval-ms", type=float,
        help="interactive delay between frames (default: derived from --fps)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="saved-animation resolution (default: 150)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display the animation even when --output is supplied",
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    if not math.isfinite(arguments.fps) or arguments.fps <= 0.0:
        parser.error("--fps must be finite and positive")
    if arguments.dpi <= 0:
        parser.error("--dpi must be positive")
    interval_ms = arguments.interval_ms
    if interval_ms is None:
        interval_ms = 1000.0 / arguments.fps
    if not math.isfinite(interval_ms) or interval_ms <= 0.0:
        parser.error("--interval-ms must be finite and positive")

    if arguments.output is not None:
        suffix = arguments.output.suffix.lower()
        if suffix not in {".gif", ".mp4"}:
            parser.error("--output must have a .gif or .mp4 extension")

    try:
        figure, animation = make_animation(arguments.csv_file, interval_ms)
        if arguments.output is not None:
            writer = "pillow" if arguments.output.suffix.lower() == ".gif" \
                else "ffmpeg"
            animation.save(
                arguments.output,
                writer=writer,
                fps=arguments.fps,
                dpi=arguments.dpi,
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    if arguments.output is None or arguments.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    main()
