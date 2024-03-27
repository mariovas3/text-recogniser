import string

from model_package.metadata import shared

RAW_DATA_DIR = shared.DATA_DIR / "raw" / "emnist"
ZIPPED_FILENAME = "matlab.zip"
TOML_FILE = RAW_DATA_DIR / "metadata.toml"
DL_DIR = shared.DATA_DIR / "downloaded" / "emnist"

PROCESSED_DATA_DIR = shared.DATA_DIR / "processed" / "emnist"
PROCESSED_DATA_FILE = "byclass.h5"
PROCESSED_FILE_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
ESSENTIALS_FILE = PROCESSED_DATA_DIR / "essentials.json"
TRAIN_FRAC = 0.8
REBALANCE_DATASET = True


NUM_SPECIAL_TOKENS = 4
INPUT_SHAPE = (28, 28)
DIMS = (1, *INPUT_SHAPE)
OUTPUT_DIMS = (1,)
MAPPING = [
    "<BLANK>",
    "<START>",
    "<END>",
    "<PAD>",
    *string.digits,
    *string.ascii_uppercase,
    *string.ascii_lowercase,
    " ",
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "?",
]
