#!/usr/bin/env python3
"""Animate applied actions and candidate rewards from a Burgers RL episode.

By default, save applied_action.mp4 and candidate_reward.mp4 beside the CSV.
Action frames use start_time; reward frames use end_time. Each CSV must contain
one episode with a fixed spatial grid and increasing decision indices.
"""

import argparse
import csv
import math
from pathlib import Path


TITLE_TIME_DECIMAL_PLACES = 3
FIELDS = {"applied_action": "start_time", "candidate_reward": "end_time"}


def make_title_formatter(frames):
    """Return a formatter whose numeric fields keep a constant width."""
    step_width = max(len(str(step)) for step, _, _, _ in frames)
    formatted_times = [
        f"{time:.{TITLE_TIME_DECIMAL_PLACES}f}"
        for _, time, _, _ in frames
    ]
    time_width = max(len(value) for value in formatted_times)

    def format_title(step, time):
        return (
            f"Decision {step:0{step_width}d}, "
            f"t = {time:0{time_width}.{TITLE_TIME_DECIMAL_PLACES}f}"
        )

    return format_title


def read_field(path, field):
    """Return ``(decision_index, time, x, values)`` tuples for all frames in *path*."""
    time_column = FIELDS[field]
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"decision_index", time_column, "x", field}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required column(s): {missing}")

        frames = []
        current_step = None
        current_time = None
        cell_centers = []
        field_values = []

        for line_number, row in enumerate(reader, start=2):
            try:
                step = int(row["decision_index"])
                time = float(row[time_column])
                cell_center = float(row["x"])
                field_value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid or missing numeric value for {field} on line {line_number}"
                ) from error

            if step < 0:
                raise ValueError(f"negative step on line {line_number}")
            if not all(math.isfinite(value) for value in
                       (time, cell_center, field_value)):
                raise ValueError(f"non-finite value on line {line_number}")

            if current_step is None:
                current_step = step
                current_time = time
            elif step != current_step or time != current_time:
                frames.append(
                    (current_step, current_time, cell_centers, field_values)
                )
                current_step = step
                current_time = time
                cell_centers = []
                field_values = []

            cell_centers.append(cell_center)
            field_values.append(field_value)

        if current_step is not None:
            frames.append(
                (current_step, current_time, cell_centers, field_values)
            )

    if not frames:
        raise ValueError("RL field history contains no data rows")

    reference_grid = frames[0][2]
    if not reference_grid:
        raise ValueError("RL field history contains an empty frame")
    previous_step = None
    previous_time = None
    for step, time, grid, _ in frames:
        if grid != reference_grid:
            raise ValueError(f"spatial grid changes at step {step}")
        if previous_step is not None and step <= previous_step:
            raise ValueError("decision indices must be strictly increasing")
        if previous_time is not None and time < previous_time:
            raise ValueError("frame times must be nondecreasing")
        previous_step = step
        previous_time = time

    return frames


def make_animation(frames, field, interval_ms):
    """Create a field animation using the appearance of animate_profiles.py."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required to animate the RL fields"
        ) from error

    all_values = [value for _, _, _, values in frames for value in values]
    minimum_value = min(all_values)
    maximum_value = max(all_values)
    value_span = maximum_value - minimum_value
    value_padding = 0.05 * value_span if value_span > 0.0 else 0.05 * max(
        1.0, abs(minimum_value)
    )

    cell_centers = frames[0][2]
    minimum_x = min(cell_centers)
    maximum_x = max(cell_centers)
    x_span = maximum_x - minimum_x
    x_padding = 0.02 * x_span if x_span > 0.0 else 0.5

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    line, = axes.plot([], [], linewidth=1.5)
    axes.set_xlim(minimum_x - x_padding, maximum_x + x_padding)
    axes.set_ylim(minimum_value - value_padding, maximum_value + value_padding)
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(field.replace("_", " ").capitalize())
    axes.grid(alpha=0.3)
    title = axes.set_title("", fontfamily="monospace")
    format_title = make_title_formatter(frames)

    def update(frame_index):
        step, time, grid, values = frames[frame_index]
        line.set_data(grid, values)
        title.set_text(format_title(step, time))
        return line, title

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    # Populate the title before laying out the figure so that saved animations
    # reserve enough space above the axes for it.
    update(0)
    figure.tight_layout()
    return figure, animation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Make separate applied-action and candidate-reward movies from burgers_rl_fields.csv."
    )
    parser.add_argument("csv_file", type=Path, help="RL field-history CSV file for one episode")
    parser.add_argument(
        "--format", choices=("mp4", "gif"), default="mp4",
        help="movie format (default: mp4; GIF uses Pillow)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="output directory (default: beside the input CSV)",
    )
    parser.add_argument(
        "--fps", type=float, default=60.0,
        help="saved-animation frame rate (default: 60)",
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
        help="display both animations after saving",
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

    output_dir = arguments.output_dir or arguments.csv_file.parent
    figures = []
    animations = []  # Keep both animations alive for interactive display.
    try:
        # Validate both fields before writing either movie.
        field_frames = {
            field: read_field(arguments.csv_file, field) for field in FIELDS
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for field, frames in field_frames.items():
            figure, animation = make_animation(frames, field, interval_ms)
            figures.append(figure)
            animations.append(animation)
            output_path = output_dir / f"{field}.{arguments.format}"
            animation.save(
                output_path,
                writer="pillow" if arguments.format == "gif" else "ffmpeg",
                fps=arguments.fps,
                dpi=arguments.dpi,
            )
            print(f"Saved {output_path}")
        if arguments.show:
            import matplotlib.pyplot as plt
            plt.show()
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    finally:
        if figures:
            import matplotlib.pyplot as plt
            for figure in figures:
                plt.close(figure)


if __name__ == "__main__":
    main()
