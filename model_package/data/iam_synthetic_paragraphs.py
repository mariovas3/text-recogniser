import random

from torch.utils.data import Dataset

import model_package.metadata.iam_synthetic_paragraphs as metadata
from model_package.data import utils
from model_package.data.create_synth_paragraphs import (
    save_iam_lines_crops_and_labels,
    synth_and_save_paragraphs,
)
from model_package.data.iam_lines import (
    load_processed_line_crops,
    load_processed_line_labels,
)
from model_package.data.iam_paragraphs import IAMParagraphs


class IAMSyntheticParagraphs(IAMParagraphs):
    def __init__(
        self,
        dataset_len=metadata.DATASET_LEN,
        min_num_lines=metadata.MIN_NUM_LINES,
        max_num_lines=metadata.MAX_NUM_LINES,
        use_precomputed=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.line_crops = None
        self.line_labels = None
        self.dataset_len = dataset_len
        self.use_precomputed = use_precomputed
        self.min_num_lines = min_num_lines
        self.max_num_lines = max_num_lines

    def prepare_data(self) -> None:
        save_iam_lines_crops_and_labels()

        # see whether to precompute paragraphs;
        if self.use_precomputed:
            print(f"Will use static synthetic paragraphs...")
            synth_and_save_paragraphs(
                size=self.dataset_len,
                min_num_lines=self.min_num_lines,
                max_num_lines=self.max_num_lines,
            )

    def setup(self, stage) -> None:
        assert stage == "fit"
        print(f"Setup Train Synthetic Paragraphs...")
        self._load_processed_crops_and_labels()
        self.train_dataset = IAMSyntheticParagraphsDataset(
            self.line_crops,
            self.line_labels,
            self.dataset_len,
            self.char_to_idx,
            self.input_dims,
            self.output_dims,
            self.trainval_transform,
            self.min_num_lines,
            self.max_num_lines,
            self.use_precomputed,
        )

    def _load_processed_crops_and_labels(self) -> None:
        if self.use_precomputed:
            data_dir = metadata.SYNTH_PAR_DIR
        else:
            data_dir = metadata.PROCESSED_DATA_DIR
        if self.line_crops is None:
            self.line_crops = load_processed_line_crops("train", data_dir)
        if self.line_labels is None:
            self.line_labels = load_processed_line_labels("train", data_dir)

    def __repr__(self):
        ans = (
            "IAMSyntheticParagraphs Dataset\n"
            f"Num classes: {len(self.idx_to_char)}\n"
            f"Input dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
            f"Paragraph Min Num Lines: {self.min_num_lines}\n"
            f"Paragraph Max Num Lines: {self.max_num_lines}\n"
        )
        return ans

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--use_precomputed",
            action="store_true",
            help="Whether to check if precomputed synth paragraphs available.",
        )


class IAMSyntheticParagraphsDataset(Dataset):
    def __init__(
        self,
        line_crops,
        line_labels,
        dataset_len,
        char_to_idx,
        input_dims,
        output_dims,
        transform,
        min_num_lines,
        max_num_lines,
        use_precomputed,
    ):
        super().__init__()
        assert len(line_crops) == len(line_labels)
        print(f"len_line_crops: {len(line_labels)}")
        self.line_crops = line_crops
        self.line_labels = line_labels
        self.ids = list(range(len(self.line_labels)))
        self.dataset_len = dataset_len
        self.char_to_idx = char_to_idx
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.transform = transform
        self.min_num_lines, self.max_num_lines = min_num_lines, max_num_lines
        self.use_precomputed = use_precomputed

    def __len__(self):
        if self.use_precomputed:
            return len(self.line_crops)
        return self.dataset_len

    def __getitem__(self, idx):
        if self.use_precomputed:
            # line crops are actually synthetic paragraphs here;
            # line labels are actually synthetic paragraphs labels here;
            datum = self.line_crops[idx]
            labels = self.line_labels[idx]
        else:
            # this whole sampling thing should be precomputed somewhere;
            num_lines = random.randint(self.min_num_lines, self.max_num_lines)
            indices = random.sample(self.ids, k=num_lines)

            while True:
                datum = utils.hstack_line_crops(
                    [self.line_crops[i] for i in indices]
                )
                labels = "\n".join([self.line_labels[i] for i in indices])
                if (
                    len(labels) <= self.output_dims[0] - 2
                    and datum.height <= self.input_dims[1]
                    and datum.width <= self.input_dims[2]
                ):
                    break
                indices = indices[:-1]
        if self.transform is not None:
            datum = self.transform(datum)
        length = self.output_dims[0]
        target = utils.convert_strings_to_labels(
            [labels],
            self.char_to_idx,
            length=length,
            with_start_and_end_tokens=True,
        )[0]
        return datum, target


if __name__ == "__main__":
    from model_package.data.testing_utils import (
        get_info_litdata,
        test_plot_para,
    )

    x, y, idx_to_char = get_info_litdata(IAMSyntheticParagraphs)
    test_plot_para(zip(x, y), len(y), idx_to_char=idx_to_char)
