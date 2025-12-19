"""User Interface module for clean, minimal ASCII output."""

import sys
from src.config import AppConfig

# ASCII Art Banner
BANNER = r"""
   __  __           _      _    _____
  |  \/  |         | |    | |  / ____|
  | \  / | ___   __| | ___| | | (___  _   _ _ __ __ _  ___ _ __ _   _
  | |\/| |/ _ \ / _` |/ _ \ |  \___ \| | | | '__/ _` |/ _ \ '__| | | |
  | |  | | (_) | (_| |  __/ |  ____) | |_| | | | (_| |  __/ |  | |_| |
  |_|  |_|\___/ \__,_|\___|_| |_____/ \__,_|_|  \__, |\___|_|   \__, |
                                                 __/ |           __/ |
                                                |___/           |___/
"""


class ConsoleUI:
    """Handles minimal console output with ASCII art and progress tracking."""

    @staticmethod
    def print_header(config: AppConfig) -> None:
        """Prints the startup banner and configuration table."""
        print(f"\033[1;36m{BANNER}\033[0m")
        print("=" * 60)
        print(" \033[1mCONFIGURATION\033[0m")
        print("-" * 60)

        # Model Details
        print(f" \033[1;33mModel:\033[0m {config.model.model_name}")
        print(f" \033[1;33mQuantization:\033[0m {config.model.quantization_bit}-bit")
        print(f" \033[1;33mDevice Map:\033[0m {config.model.device_map}")

        # Surgery Details
        print("-" * 60)
        print(f" \033[1;33mEnergy Threshold:\033[0m {config.surgery.energy_threshold}")
        print(
            f" \033[1;33mTarget Modules:\033[0m {config.surgery.target_modules or 'All Linear'}"
        )

        # Paths
        print("-" * 60)
        print(f" \033[1;33mOutput Dir:\033[0m {config.paths.output_dir}")
        print("=" * 60)
        print("\n")

    @staticmethod
    def progress_bar(
        iterable: object,
        total: int,
        prefix: str = "Progress",
        length: int = 40,
        fill: str = "█",
    ) -> object:
        """Simple ASCII progress bar generator."""
        # Force TTY behavior if desired, but relying on is_tty is safer.
        # However, to ensure minimal updates without newlines in notebooks/consoles:
        # We don't need to assign is_tty if we are always assuming it for this bar.

        def print_bar(iteration: int, suffix: str = "") -> None:
            percent = ("{0:.1f}").format(100 * (iteration / float(total)))
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + "-" * (length - filled_length)
            # Clear line and print
            sys.stdout.write(f"\r\033[K{prefix} |{bar}| {percent}% {suffix}")
            sys.stdout.flush()

        print_bar(0)
        # Type ignore because we know iterable is iterable but mypy is strict
        for i, item in enumerate(iterable):  # type: ignore
            yield item
            print_bar(i + 1)

        print()  # Newline on complete

    @staticmethod
    def status_update(message: str) -> None:
        """Prints a transient status update."""
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K> {message}")
            sys.stdout.flush()
        else:
            print(f"> {message}")
