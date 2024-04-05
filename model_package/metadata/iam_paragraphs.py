from model_package.metadata import emnist, shared

PROCESSED_DATA_DIR = shared.DATA_DIR / "processed" / "iam_paragraphs"

# add new line token;
MAPPING = [*emnist.MAPPING, "\n"]

IMAGE_SCALE_FACTOR = 2
# if 28 pixels per line and the max line count is 13;
# 28 * 13 = 364;
IMAGE_HEIGHT = 400
# 600 > num_lines * max_aspect_ratio * 28
IMAGE_WIDTH = 600

# for the cropping transforms;
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# empirical max char count in paragraphs was 680;
# adding start and end token gives 682;
MAX_PAR_LEN = 682
DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (MAX_PAR_LEN,)
