from model_package.metadata import shared

RAW_DATA_DIR = shared.DATA_DIR / "raw" / "iam"
DL_DIR = shared.DATA_DIR / "downloaded" / "iam"
EXTRACTED_DATA_DIR = DL_DIR / "iamdb"
TOML_FILE = RAW_DATA_DIR / "metadata.toml"

ZIPPED_FILENAME = "fsdl_downsampled_iam_forms.zip"
