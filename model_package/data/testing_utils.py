import json
from pathlib import Path

# to root dir;
p = Path(__file__).absolute().parents[2]
TESTING_OUTPUTS_DIR = p / "testing_outputs"
IAM_PLOTS = TESTING_OUTPUTS_DIR / "iam" / "plots"
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model_package.data.iam import IAM


def check_line_count_dist(labels_path):
    with open(labels_path, "r") as file:
        labels = json.load(file)
    counts = [label.count("\n") + 1 for label in labels]
    counts = [_.count("\n") + 1 for _ in labels]
    xs, ys = np.unique(counts, return_counts=True)
    plt.bar(xs, ys)
    plt.xticks(xs, xs)
    plt.title("line count dist")
    plt.show()


def test_plot_para(data_batch, batch_size, idx_to_char):
    fig = plt.figure(figsize=(12, 12))
    to_show = min(9, batch_size)
    for i in range(to_show):
        x, y = next(data_batch)
        plt.subplot(3, 3, i + 1)
        plt.imshow(x.squeeze().numpy(), cmap="gray")
        num_lines = sum(idx_to_char[idx.item()] == "\n" for idx in y) + 1
        plt.title(f"{num_lines =}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def test_plot_lines(data_batch, batch_size, idx_to_char, min_idx=0):
    to_show = min(10, batch_size)
    for i in range(to_show):
        x, y = next(data_batch)
        plt.subplot(10, 1, i + 1)
        plt.imshow(x.squeeze().numpy(), cmap="gray")
        label = "".join(
            [idx_to_char[idx.item()] for idx in y if idx >= min_idx]
        )
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_info_litdata(lit_datamodule):
    import random

    import numpy as np
    import torch

    from model_package.data.lit_datamodule import mock_lit_dataset

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    now = time.time()
    dataset, args = mock_lit_dataset(lit_datamodule)
    dataset.prepare_data()
    dataset.setup(args["stage"])
    print(dataset)
    if args["stage"] == "fit":
        dl = iter(dataset.train_dataloader())
    else:
        dl = iter(dataset.test_dataloader())
    x, y = next(dl)
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(
        f"x dtype, min, mean, max, std: {(x.dtype, x.min(), x.mean(), x.max(), x.std())}"
    )
    print(f"y dtype, min, max: {(y.dtype, y.min(), y.max())}")
    print(f"batch_size: {len(y)}")
    print(f"took time: {time.time() - now:.5f}s")
    return x, y, dataset.idx_to_char


def crop_img(img: Image, region: dict):
    """Return cropped region of img defined in region."""
    return img.crop((region["x1"], region["y1"], region["x2"], region["y2"]))


def get_max_dims_lines(iam: IAM):
    ans_width = {"file_id": None, "line_idx": None}
    ans_height = {"file_id": None, "line_idx": None}
    max_width, max_height = None, None
    for file_id in iam.line_regions_by_id:
        for i, region in enumerate(iam.line_regions_by_id[file_id]):
            temp_x = region["x2"] - region["x1"]
            temp_y = region["y2"] - region["y1"]
            if not max_width or temp_x > max_width:
                max_width = temp_x
                ans_width["file_id"] = file_id
                ans_width["line_idx"] = i
            if not max_height or temp_y > max_height:
                max_height = temp_y
                ans_height["file_id"] = file_id
                ans_height["line_idx"] = i
    ans_width["max_width"] = max_width
    ans_height["max_height"] = max_height
    return ans_width, ans_height


def get_min_dims_lines(iam: IAM):
    ans_width = {"file_id": None, "line_idx": None}
    ans_height = {"file_id": None, "line_idx": None}
    min_width, min_height = None, None
    for file_id in iam.line_regions_by_id:
        for i, region in enumerate(iam.line_regions_by_id[file_id]):
            temp_x = region["x2"] - region["x1"]
            temp_y = region["y2"] - region["y1"]
            if not min_width or temp_x < min_width:
                min_width = temp_x
                ans_width["file_id"] = file_id
                ans_width["line_idx"] = i
            if not min_height or temp_y < min_height:
                min_height = temp_y
                ans_height["file_id"] = file_id
                ans_height["line_idx"] = i
    ans_width["min_width"] = min_width
    ans_height["min_height"] = min_height
    return ans_width, ans_height


def get_heights_widths(iam: IAM):
    heights = []
    widths = []
    for _ in iam.line_regions_by_id:
        for region in iam.line_regions_by_id[_]:
            heights.append(region["y2"] - region["y1"])
            widths.append(region["x2"] - region["x1"])
    return heights, widths


def quick_imshow(img: Image, region: dict, plot_name: str):
    """Ideally plot_name should contain the form_id."""
    plt.imshow(crop_img(img, region), cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(IAM_PLOTS / plot_name)
    plt.close()


def hist_of_linecount_per_form(iam: IAM):
    # visualise dist of num valid lines per form;
    lens = [len(_) for _ in iam.line_strings_by_id.values()]
    plt.hist(lens)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(IAM_PLOTS / "linecount_per_form_hist.png")
    plt.close()
