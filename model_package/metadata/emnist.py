import string

from model_package.metadata import shared

RAW_DATA_DIR = shared.DATA_DIR / "raw" / "emnist"
RAW_FILENAME = "matlab.zip"
# THEY BROKE THE LINK ON THEIR WEBSITE!!!
# THIS NO LONGER LEADS TO THE ZIP FILE BUT REDIRECTS
# TO SOME HTML; I STILL HAVE THE ZIP LOCALLY AND WILL
# TRY TO UPLOAD IT SOMEWHERE THAT IS EASY TO DOWNLOAD.
# TODO: change the RAW_URL;
RAW_URL = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
PROCESSED_DATA_DIR = shared.DATA_DIR / "processed" / "emnist"
PROCESSED_DATA_FILE = "byclass.h5"
PROCESSED_FILE_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE
ESSENTIALS_FILE = PROCESSED_DATA_DIR / "essentials.json"
TRAIN_FRAC = 0.8
TRUNCATE_DATASET = True


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
