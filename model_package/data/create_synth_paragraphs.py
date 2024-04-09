import random
from argparse import ArgumentParser

import model_package.metadata.iam_synthetic_paragraphs as metadata
from model_package.data import utils
from model_package.data.iam_lines import save_images_and_labels
from model_package.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs
from model_package.metadata.iam_paragraphs import DIMS, OUTPUT_DIMS


def main():
    args = get_args()
    iam_synth = IAMSyntheticParagraphs(**args)
    # saves line crops and labels from IAM dataset;
    iam_synth.prepare_data()
    iam_synth._load_processed_crops_and_labels()
    paragraphs, labels = get_paragraphs_and_labels(args, iam_synth)
    save_images_and_labels(
        paragraphs,
        labels,
        split="train",
        dir_path=metadata.PROCESSED_DATA_DIR / "synth_paragraphs",
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


def get_paragraphs_and_labels(args, iam_synth: IAMSyntheticParagraphs):
    assert (
        iam_synth.line_crops is not None
    ), "Need to load the crops and labels."
    size = args.get("size", iam_synth.dataset_len)
    min_num_lines = args.get("min_num_lines", 1)
    max_num_lines = args.get("max_num_lines", 13)
    seed = args.get("seed", 0)
    ids = list(range(len(iam_synth.line_crops)))

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
        while indices:
            datum = utils.hstack_line_crops(
                [iam_synth.line_crops[i] for i in indices]
            )
            labels = "\n".join([iam_synth.line_labels[i] for i in indices])
            if (
                len(labels) <= OUTPUT_DIMS[0] - 2
                and datum.height <= DIMS[1]
                and datum.width <= DIMS[2]
            ):
                break
            indices = indices[:-1]
        if not indices:
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
        default=1,
        help="Minimum number of lines in synthetic paragraph.",
    )
    parser.add_argument(
        "--max_num_lines",
        type=int,
        default=13,
        help="Maximum number of lines in synthetic paragraph.",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
