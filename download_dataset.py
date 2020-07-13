import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_dataset():
    filepath = './batch_data.zip'
    if os.path.exists(filepath):
        print('{} already exists!'.format(filepath))
    else:
        gdd.download_file_from_google_drive(file_id='1rBONTkSee9xr2DnqK9eTjGE5_Z71Fvl6', dest_path=filepath, unzip=True, showsize=True)

if __name__ == "__main__":
    download_dataset()
