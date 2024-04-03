from model_package.metadata import emnist, shared

PROCESSED_DATA_DIR = shared.DATA_DIR / "processed" / "iam_lines"

IMAGE_SCALE_FACTOR = 2

# pick image height and image width such that
# image_height * max_aspect_raio <= image_width;
# so that I can use resize_crop on the image and pad
# to (image_height, image_width) dims;
# the aspect ratio is defined as width divided by height;

# given a max aspect ratio of slightly less than 28,
# we can set IMAGE_WIDTH=800 and IMAGE_HEIGHT=28
# in the metadata;
# I chose 28 bc that's similar to the NIST data;
IMAGE_HEIGHT = 28  # input empirical max is 119;
IMAGE_WIDTH = 800  # input emirical max is 1151;

DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
# emirically, I checked max character handwritten line is 87;
# max(len(v2) for v1 in iam.line_strings_by_id.values() for v2 in v1)
# accounting for start and end token we get 89 chars;
OUTPUT_DIMS = (89,)

MAPPING = emnist.MAPPING
