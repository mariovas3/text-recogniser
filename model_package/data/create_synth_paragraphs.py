import random
from argparse import ArgumentParser

import model_package.metadata.iam_synthetic_paragraphs as metadata
from model_package.data import utils
from model_package.data.iam import IAM
from model_package.data.iam_lines import (
    generate_line_crops_and_labels,
    load_processed_crops_and_labels,
    save_images_and_labels,
)
from model_package.metadata.iam_paragraphs import DIMS, OUTPUT_DIMS


def _is_synth_saved():
    if metadata.SYNTH_PAR_DIR.exists():
        try:
            counts = len(
                list((metadata.SYNTH_PAR_DIR / "train").glob("*.png"))
            )
        except:
            return False
        print(f"{counts} synth paragraphs found...")
        return counts > 0
    return False


def save_iam_lines_crops_and_labels():
    if metadata.PROCESSED_DATA_DIR.exists():
        print(f"PROCESSED DIR already exists: {metadata.PROCESSED_DATA_DIR}")
        return
    print(f"Preparing IAMSyntheticParagraphs...")
    print(f"Processing IAMLines crops and labels...")
    iam = IAM()
    iam.prepare_data()

    crops, labels = generate_line_crops_and_labels(iam, "train")
    save_images_and_labels(crops, labels, "train", metadata.PROCESSED_DATA_DIR)


def synth_and_save_paragraphs(
    size=metadata.DATASET_LEN,
    seed=0,
    min_num_lines=metadata.MIN_NUM_LINES,
    max_num_lines=metadata.MAX_NUM_LINES,
):
    save_iam_lines_crops_and_labels()
    if not _is_synth_saved():
        print(f"Synthesizing paragraphs...")
        line_crops, line_labels = load_processed_crops_and_labels(
            "train", metadata.PROCESSED_DATA_DIR
        )
        args = {
            "size": size,
            "seed": seed,
            "min_num_lines": min_num_lines,
            "max_num_lines": max_num_lines,
        }
        paragraphs, labels = get_paragraphs_and_labels(
            args, line_crops, line_labels
        )
        save_images_and_labels(
            paragraphs,
            labels,
            split="train",
            dir_path=metadata.SYNTH_PAR_DIR,
        )


class LineIdxGenerator:
    def __init__(
        self, size, ids: list[int], min_lines, max_lines, seed
    ) -> None:
        random.seed(seed)
        self.num_lines = random.choices(
            range(min_lines, max_lines + 1), k=size
        )
        self.indices_ptr = 0
        self.num_lines_ptr = 0
        self.indices = ids.copy()
        random.shuffle(self.indices)

    def sample_line_idxs(self):
        num_lines = self.num_lines[self.num_lines_ptr]
        self.num_lines_ptr += 1

        ans = []
        diff = num_lines
        while self.indices_ptr + diff >= len(self.indices):
            # replicated lines in a single paragraph are possible
            # due to taking what's left and then generating new
            # sequence of idxs where the new prefix may contain
            # elements of the old suffix;
            ans.extend(self.indices[self.indices_ptr :])
            self.indices_ptr = 0
            random.shuffle(self.indices)
            diff = num_lines - len(ans)
        ans.extend(self.indices[self.indices_ptr : self.indices_ptr + diff])
        self.indices_ptr += diff
        return ans


def get_paragraphs_and_labels(args, line_crops, line_labels):
    size = args.get("size", metadata.DATASET_LEN)
    min_num_lines = args.get("min_num_lines", metadata.MIN_NUM_LINES)
    max_num_lines = args.get("max_num_lines", metadata.MAX_NUM_LINES)
    seed = args.get("seed", 0)
    ids = list(range(len(line_crops)))

    # get line idx sampler;
    line_idx_gen = LineIdxGenerator(
        size,
        ids,
        min_num_lines,
        max_num_lines,
        seed,
    )
    paragraphs, label_strings = [], []

    # synthesize the paragraphs;
    max_iter = round(size * 1.1)
    counter = 0
    invalid_count = 0
    while len(paragraphs) < size and counter < max_iter:
        indices = line_idx_gen.sample_line_idxs()
        counter += 1
        while len(indices) >= min_num_lines:
            datum = utils.hstack_line_crops([line_crops[i] for i in indices])
            labels = "\n".join([line_labels[i] for i in indices])
            if (
                len(labels) <= OUTPUT_DIMS[0] - 2
                and datum.height <= DIMS[1]
                and datum.width <= DIMS[2]
            ):
                break
            indices = indices[:-1]
        if not indices or len(indices) < min_num_lines:
            invalid_count += 1
            line_idx_gen.num_lines_ptr -= 1
            continue
        paragraphs.append(datum)
        label_strings.append(labels)
    print(f"invalid synthesis count: {invalid_count}")
    print(f"num generated: {len(paragraphs)}/{size}")
    return paragraphs, label_strings


def get_args() -> dict:
    parser = ArgumentParser(
        description="Create and save Iam Synthetic Paragraphs.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=metadata.DATASET_LEN,
        help="Set size of created dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for generating the synthetic paragraphs.",
    )
    parser.add_argument(
        "--min_num_lines",
        type=int,
        default=metadata.MIN_NUM_LINES,
        help=f"Minimum number of lines in synthetic paragraph; default is {metadata.MIN_NUM_LINES}",
    )
    parser.add_argument(
        "--max_num_lines",
        type=int,
        default=metadata.MAX_NUM_LINES,
        help=f"Maximum number of lines in synthetic paragraph; default is {metadata.MAX_NUM_LINES}",
    )
    return vars(parser.parse_args())


def main():
    args = get_args()
    synth_and_save_paragraphs(**args)


if __name__ == "__main__":
    main()
