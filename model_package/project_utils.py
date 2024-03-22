import io
import os
from contextlib import contextmanager
from pathlib import Path

# parent of model_package where the google drive api credintials should be;
p = Path(__file__).absolute().parent.parent
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def download_from_google_drive(
    file_id: str, save_destination_dir: Path, save_as_filename: str
):
    """
    Download a file from google drive using the file_id.

    Args:
        file_id: Id of the file to download. Can get it by
            copying the share link of the file in the google
            drive GUI and the long sequence of chars is the id.
        save_destination_dir: the directory where to download.
        save_as_filename: the name of the file in the destination dir.
    Returns: Path object to downloaded file destination.
    """
    # skip if file already exists;
    save_destination_dir.mkdir(parents=True, exist_ok=True)
    full_name = save_destination_dir / save_as_filename
    if full_name.exists():
        print(f"\nFILE ALREADY EXISTS!\n{full_name}")
        return full_name
    # download from google drive otherwise;
    creds = None
    # set scope to read only or download files;
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    # token credentials should be in root dir if present;
    token_path = p / "token.json"
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # if there are no (valid) credentials available, let
    # the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("refreshing google drive credentials...")
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                p / "google_drive_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run;
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    # create google drive api client and download file;
    try:
        with build("drive", "v3", credentials=creds) as service:
            request = service.files().get_media(fileId=file_id)
            file = io.BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            print(f"Downlading file...")
            while done is False:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%")
    except HttpError as error:
        print(f"An error occurred during download: {error}")
        file = None
    with open(full_name, "wb") as f:
        f.write(file.getvalue())
    print(f"file downloaded successfully to {full_name}")
    return full_name


@contextmanager
def change_wd(to_dir):
    curdir = os.getcwd()
    try:
        os.chdir(to_dir)
        yield
    finally:
        os.chdir(curdir)


def download_raw_dataset(url, save_destination_dir, save_as_filename):
    # skip mkdir if exists, otherwise also create parents;
    save_destination_dir.mkdir(parents=True, exist_ok=True)
    full_name = save_destination_dir / save_as_filename
    if full_name.exists():
        print(f"\nFILE ALREADY EXISTS!\n{full_name}")
        return full_name
    response = requests.get(
        url,
    )
    if response.status_code == 200:
        with open(full_name, "wb") as file:
            file.write(response.content)
        print(f"file downloaded successfully to {full_name}")
        return full_name
    print(
        f"Failed download of {full_name}! status code: {response.status_code}"
    )
    return full_name
