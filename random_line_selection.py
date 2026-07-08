import argparse
import random
from pathlib import Path


def sample_lines_from_file(input_file: Path, output_file: Path, n_lines: int):
    # Read all non-empty lines
    with input_file.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        print(f"[WARNING] Empty file skipped: {input_file.name}")
        return

    # If file has fewer lines than requested, take all lines
    k = min(n_lines, len(lines))

    sampled = random.sample(lines, k)

    # Write sampled lines
    with output_file.open("w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    print(f"[OK] {input_file.name} -> {output_file.name} ({k} lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample lines from txt files."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .txt files",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where sampled files will be written",
    )

    parser.add_argument(
        "num_lines",
        type=int,
        help="Number of random lines to select per file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_dir.glob("*"))

    if not txt_files:
        print("No .txt files found.")
        return
    i=0
    for txt_file in txt_files:
        i += 1
        output_file = output_dir / txt_file.name
        sample_lines_from_file(txt_file, output_file, args.num_lines)
        if i % 10 == 0:
            print(f"Processed {i}/{len(txt_files)} files...")



if __name__ == "__main__":
    main()