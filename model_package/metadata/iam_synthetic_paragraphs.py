from model_package.metadata import shared

PROCESSED_DATA_DIR = shared.DATA_DIR / "processed" / "iam_synthetic_paragraphs"
SYNTH_PAR_DIR = PROCESSED_DATA_DIR / "synth_paragraphs"
MIN_NUM_LINES, MAX_NUM_LINES = 1, 13

EXPECTED_BATCH_SIZE = 64
EXPECTED_GPUS = 8
EXPECTED_STEPS = 40

# set the dataset's length based on parameters during typical training
DATASET_LEN = EXPECTED_BATCH_SIZE * EXPECTED_GPUS * EXPECTED_STEPS
