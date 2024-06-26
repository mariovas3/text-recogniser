from pathlib import Path

import toml
from boltons.cacheutils import cachedproperty
from defusedxml import ElementTree
from PIL import ImageOps

import model_package.metadata.iam as metadata
from model_package.data import utils
from model_package.project_utils import download_from_google_drive


class IAM:
    """
    Images are identified by their "form ID". These IDs
    are used to separate train, validation and test splits,
    as keys for dictonaries returning label and
    image crop region data, and more.

    We care about extracting line-level data from
    the IAM forms. Based on these lines we are also able to build
    paragraphs, which is what we ultimately train on.

    There are a bunch of issues and quirks with extracting lines,
    better described in the project README.

    In summary:
        - I added a heuristic for limiting the line height
            otherwise we can get images that capture two
            lines but are matched with a text label only
            for the first line.
        - Some handwritten text is incorectly copied. E.g., in form
            p02-109 the author has also included the
            'Sentence Database         P02-109' header and has
            drawn horizontal lines that surround the actual machine part
            that they were supposed to write. To deal with this,
            I get lines starting from the first line that occurred in
            the machine part and ending at the first line from bottom
            to top that occurred in the machine part.
            - Since the machine part is itself inconsistent in its
                use of hyphons at the end of machine lines, I don't
                do a strict check for all handwritten lines to occur
                in the machine part. The assumption is that the first
                handwritten ones should occur though.
        - In some forms (5 in total), e.g., g07-000a, the author has
            written in all caps. This doesn't match the given machine
            part annotation, but I just uppercase it and procede as
            for the other forms.
            - All 5 all-caps forms are in the training set.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.toml_metadata = toml.load(metadata.TOML_FILE)

    @property
    def xml_filenames(self) -> list[Path]:
        """A list of paths to .xml files."""
        return list((metadata.EXTRACTED_DATA_DIR / "xml").glob("*.xml"))

    def prepare_data(self) -> None:
        if self.xml_filenames:
            print("XML FILES FOR IAM FOUND! SKIPPING prepare_data call...")
            return
        zipped_file = download_from_google_drive(
            self.toml_metadata["google_drive_id"],
            metadata.DL_DIR,
            self.toml_metadata["filename"],
        )
        utils.extract_raw_dataset(metadata.ZIPPED_FILENAME, metadata.DL_DIR)

    @property
    def form_filenames(self):
        """A list of paths to form .jpg files."""
        return list((metadata.EXTRACTED_DATA_DIR / "forms").glob("*.jpg"))

    @property
    def form_filenames_by_id(self):
        """A dictionary from form id to path of form."""
        # f is path/to/form/formname.jpg
        # f.stem is f"{formname}"
        return {f.stem: f for f in self.form_filenames}

    def load_image(self, form_id):
        image = utils.read_image_pil(
            self.form_filenames_by_id[form_id], grayscale=True
        )
        return ImageOps.invert(image)

    @cachedproperty
    def machine_string_by_id(self):
        return {
            f.stem: _get_machine_string_from_xml(f) for f in self.xml_filenames
        }

    @cachedproperty
    def all_ids(self) -> set[str]:
        """
        A set of all non-ignored form IDs.

        should be at most:
        ls iamdb/forms -1 | wc -l
        which is 1539; ids of bad forms are ignored.
        """
        return {file_id for file_id in self.line_strings_by_id}

    @cachedproperty
    def test_ids(self) -> set[str]:
        """
        should be at most:
        cat iamdb/task/testset.txt | cut -f 1,2 -d - | uniq | wc -l
        which is 232 since truncate the last '-'-separated element;
        otherwise `wc -l iamdb/task/testset.txt` gives 1861 as per lwitlrt task;
        """
        ans = _get_form_ids(metadata.EXTRACTED_DATA_DIR / "task/testset.txt")
        return ans.intersection(self.all_ids)

    @cachedproperty
    def validation_ids(self) -> set[str]:
        val_ids = _get_form_ids(
            metadata.EXTRACTED_DATA_DIR / "task/validationset1.txt"
        )
        val_ids.union(
            _get_form_ids(
                metadata.EXTRACTED_DATA_DIR / "task/validationset2.txt"
            )
        )
        return val_ids.intersection(self.all_ids)

    @cachedproperty
    def train_ids(self) -> set[str]:
        return self.all_ids.difference(
            self.validation_ids.union(self.test_ids)
        )

    @cachedproperty
    def ids_by_split(self) -> dict[str, set]:
        return {
            "train": self.train_ids,
            "val": self.validation_ids,
            "test": self.test_ids,
        }

    @cachedproperty
    def split_by_id(self):
        """A dictionary mapping form page_ids to their split according to IAM Lines LWITLRT."""
        ans = {page_id: "train" for page_id in self.train_ids}
        ans.update({page_id: "test" for page_id in self.test_ids})
        ans.update({page_id: "val" for page_id in self.validation_ids})
        return ans

    @cachedproperty
    def line_strings_by_id(self):
        """
        Return a dict from file id to list of strings of handwritten lines.

        Start from first line that occurrs in the relevant machine
        part and end at first line, bottom to top, that occurs in
        the machine part.
        """
        ans = {}
        for f in self.xml_filenames:
            validation_string = self.machine_string_by_id[f.stem]
            # uppercase the validation string for the all caps forms;
            if f.stem in metadata.ALL_CAPS_FORM_IDS:
                validation_string = validation_string.upper()
            valid_line_strings = _get_line_strings_from_xml_file(
                f, validation_string
            )
            if valid_line_strings:
                ans[f.stem] = valid_line_strings
        return ans

    @cachedproperty
    def paragraph_string_by_id(self):
        return {
            form_id: "\n".join(lines)
            for form_id, lines in self.line_strings_by_id.items()
        }

    @cachedproperty
    def line_regions_by_id(self):
        """
        Return a dict of form_id to list of dicts of x1,y1,x2,y2 coords of lines.

        Start from first line that occurs in the relevant machine
        part and end at first line, bottom to top, that occurs in
        the machine part.
        """
        ans = {}
        for f in self.xml_filenames:
            validation_string = self.machine_string_by_id[f.stem]
            # make validation string all caps for all-caps forms;
            if f.stem in metadata.ALL_CAPS_FORM_IDS:
                validation_string = validation_string.upper()
            valid_line_regions = _get_line_regions_from_xml_file(
                f, validation_string
            )
            if valid_line_regions:
                ans[f.stem] = valid_line_regions
        return ans

    @cachedproperty
    def paragraph_regions_by_id(self):
        ans = {}
        for form_id, line_coords in self.line_regions_by_id.items():
            for i, coords in enumerate(line_coords):
                if i == 0:
                    # copies a dict[str, int] so no need for deep copies;
                    ans[form_id] = coords.copy()
                else:
                    ans[form_id]["x1"] = min(ans[form_id]["x1"], coords["x1"])
                    ans[form_id]["y1"] = min(ans[form_id]["y1"], coords["y1"])
                    ans[form_id]["x2"] = max(ans[form_id]["x2"], coords["x2"])
                    ans[form_id]["y2"] = max(ans[form_id]["y2"], coords["y2"])
        return ans

    def __repr__(self):
        info = ["IAM Dataset Info:"]
        info.append(f"Total Images: {len(self.all_ids)}")
        info.append(f"Total Train Images: {len(self.train_ids)}")
        info.append(f"Total Val Images: {len(self.validation_ids)}")
        info.append(f"Total Test Images: {len(self.test_ids)}")
        # at most same as num forms - 1539;
        info.append(f"Total Paragraphs: {len(self.paragraph_string_by_id)}")
        # should give at most 13353, same as in the "Characteristics" section
        # on the IAM website;
        # in out case we ignore bad lines so might get fewer lines;
        num_lines = sum(
            len(lines) for _, lines in self.line_strings_by_id.items()
        )
        info.append(f"Total Lines: {num_lines}")
        return "\n\t".join(info)


def _get_machine_string_from_xml(xml_file):
    root = ElementTree.parse(xml_file).getroot()
    machine_lines = root.findall("machine-printed-part/machine-print-line")
    return " ".join([el.attrib["text"] for el in machine_lines])


def _get_coords_of_xml_element(xml_element, xml_path):
    xml_subelements = xml_element.findall(xml_path)
    if not xml_subelements:
        raise ValueError(f"{xml_path} not found in xml_element")

    ans = {
        "x1": None,
        "y1": None,
        "x2": None,
        "y2": None,
    }
    for i, el in enumerate(xml_subelements):
        if i == 0:
            ans["x1"] = int(el.attrib["x"])
            ans["y1"] = int(el.attrib["y"])
            ans["x2"] = int(el.attrib["x"]) + int(el.attrib["width"])
            # limit the height of the line;
            ans["y2"] = min(
                int(el.attrib["y"]) + int(el.attrib["height"]),
                ans["y1"] + 2 * metadata.MAX_LINE_HEIGHT,
            )
        else:
            ans["x1"] = min(ans["x1"], int(el.attrib["x"]))
            ans["y1"] = min(ans["y1"], int(el.attrib["y"]))
            ans["x2"] = max(
                ans["x2"], int(el.attrib["x"]) + int(el.attrib["width"])
            )
            ans["y2"] = max(
                ans["y2"],
                # found some issues with overlapping lines e.g.,
                # in file_id 'r02-060'
                # line at idx 8 overlaps with line at idx 9;
                min(
                    int(el.attrib["y"]) + int(el.attrib["height"]),
                    ans["y1"] + 2 * metadata.MAX_LINE_HEIGHT,
                ),
            )
    # downsample accordingly;
    # the assert below is not guaranteed because y1 can decrease
    # its value, but y2 can only increase its value, so for a fixed
    # y2, we can increase effective height by decreasing y1;
    # in essence this MAX_LINE_HEIGHT trick only protects us
    # from overshooting due to el.attrib[y] + el.attrib[height]
    # in buggy annotations of forms;
    # assert ans["y2"] - ans["y1"] <= 2 * metadata.MAX_LINE_HEIGHT, f"y2: {ans['y2']}, y1: {ans['y1']}"
    return {
        key: value // metadata.DOWNSAMPLE_FACTOR for key, value in ans.items()
    }


def _get_hw_line_xml_elements(xml_file_path):
    root = ElementTree.parse(xml_file_path).getroot()
    return root.findall("handwritten-part/line")


def _get_line_regions_from_xml_file(xml_file_path, valid_string) -> list[dict]:
    xml_line_elements = _get_hw_line_xml_elements(xml_file_path)
    start, end = find_start_and_end_line(
        xml_file_path, valid_string, xml_line_elements
    )
    if start is None:
        return []
    coords_of_lines = [
        _get_coords_of_xml_element(xml_line_elements[k], "word/cmp")
        for k in range(start, end + 1)
    ]
    # next_line_region["y1"] - prev_line_region["y2"] < 0 possible due to overlapping characters
    line_gaps_y = [
        max(next_line_region["y1"] - prev_line_region["y2"], 0)
        for next_line_region, prev_line_region in zip(
            coords_of_lines[1:], coords_of_lines[:-1]
        )
    ]
    post_line_gaps_y = line_gaps_y + [2 * metadata.LINE_REGION_PADDING]
    pre_line_gaps_y = [2 * metadata.LINE_REGION_PADDING] + line_gaps_y

    return [
        {
            "x1": coords["x1"] - metadata.LINE_REGION_PADDING,
            "x2": coords["x2"] + metadata.LINE_REGION_PADDING,
            "y1": coords["y1"]
            - min(metadata.LINE_REGION_PADDING, pre_line_gaps_y[i] // 2),
            "y2": coords["y2"]
            + min(metadata.LINE_REGION_PADDING, post_line_gaps_y[i] // 2),
        }
        for i, coords in enumerate(coords_of_lines)
    ]


def _get_line_strings_from_xml_file(xml_file_path, valid_string) -> list[str]:
    """Return list of strings for each handwritten line in a form."""
    xml_line_elements = _get_hw_line_xml_elements(xml_file_path)
    start, end = find_start_and_end_line(
        xml_file_path, valid_string, xml_line_elements
    )
    if start is None:
        return []
    return [
        xml_line_elements[k].attrib["text"].replace("&quot;", '"')
        for k in range(start, end + 1)
    ]


def find_start_and_end_line(xml_file, valid_string, xml_line_elements=None):
    """Get tuple of start and end indexes."""
    if xml_line_elements is None:
        xml_line_elements = _get_hw_line_xml_elements(xml_file)
    found_first = False
    for i in range(len(xml_line_elements)):
        curr = xml_line_elements[i].attrib["text"]
        if curr in valid_string or curr[:-1] in valid_string:
            found_first = True
            break

    if not found_first:
        return None, None

    for j in range(len(xml_line_elements) - 1, -1, -1):
        curr = xml_line_elements[j].attrib["text"]
        if curr in valid_string or curr[:-1] in valid_string:
            return i, j
    return i, j


def _get_form_ids(split_filepath) -> set[str]:
    with open(split_filepath, "r") as f:
        line_ids = f.read().split("\n")
    page_ids = set()
    for line_id in line_ids:
        if line_id:
            temp = "-".join(line_id.split("-")[:2])
            if not temp in page_ids:
                page_ids.add(temp)
    return page_ids


if __name__ == "__main__":
    iam = IAM()
    iam.prepare_data()
    print(iam)
