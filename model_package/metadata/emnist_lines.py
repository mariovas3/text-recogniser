from model_package.metadata import emnist

PROCESSED_DATA_DIR = emnist.PROCESSED_DATA_DIR.parent / "emnist_lines"
ESSENTIALS_FILE = (
    emnist.ESSENTIALS_FILE.parent / "emnist_lines_essentials.json"
)

MAPPING = emnist.MAPPING

CHAR_HEIGHT, CHAR_WIDTH = 28, 28
