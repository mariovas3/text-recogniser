import zipfile
from pathlib import Path
from typing import List

import toml

import model_package.metadata.iam as metadata
from model_package.data.lit_datamodule import BaseDataModule
from model_package.project_utils import change_wd, download_from_google_drive


class IAM(BaseDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.toml_metadata = toml.load(metadata.TOML_FILE)

    @property
    def xml_filenames(self) -> List[Path]:
        """A list of the filenames of all .xml files, which contain label information."""
        return list((metadata.EXTRACTED_DATA_DIR / "xml").glob("*.xml"))

    def prepare_data(self) -> None:
        if self.xml_filenames:
            return
        zipped_file = download_from_google_drive(
            self.toml_metadata["google_drive_id"],
            metadata.DL_DIR,
            self.toml_metadata["filename"],
        )
        _extract_raw_dataset(metadata.ZIPPED_FILENAME, metadata.DL_DIR)


def _extract_raw_dataset(filename, dirname):
    print(f"Extracting {filename}...")
    with change_wd(dirname):
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extractall()
