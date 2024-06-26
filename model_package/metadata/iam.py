from model_package.metadata import shared

RAW_DATA_DIR = shared.DATA_DIR / "raw" / "iam"
DL_DIR = shared.DATA_DIR / "downloaded" / "iam"
EXTRACTED_DATA_DIR = DL_DIR / "iamdb"
TOML_FILE = RAW_DATA_DIR / "metadata.toml"

ZIPPED_FILENAME = "fsdl_downsampled_iam_forms.zip"
# images were downsampled so should take that into account
# when processing the xml info;
DOWNSAMPLE_FACTOR = 2
# add this many pixels around the exact coordinates;
LINE_REGION_PADDING = 8
MAX_LINE_HEIGHT = 60

# these are the total counts if no filtering done;
# TOTAL_IMAGES = 1539
# TOTAL_TEST_IMAGES = 232
# TOTAL_PARAGRAPHS = 1539
# TOTAL_LINES = 13353

# total counts after filtering out bad lines and/or forms;
TOTAL_IMAGES = 1_539
TOTAL_TRAIN_IMAGES = 1_202
TOTAL_VAL_IMAGES = 105
TOTAL_TEST_IMAGES = 232
TOTAL_PARAGRAPHS = 1_539
TOTAL_LINES = 13_287

# these are forms in which the author wrote in all caps;
# all of them are in the training set;
ALL_CAPS_FORM_IDS = ["g07-000a", "g07-003a", "f07-101a", "g07-007a", "g01-022"]
