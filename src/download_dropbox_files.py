# src/download_dropbox_files.py

import dropbox
import os
import requests
import sys

# Function to refresh the access token
def refresh_access_token(refresh_token, client_id, client_secret):
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"❌ Failed to refresh access token: {response.text}")

# Function to download ALL files (no extension filtering) and delete them afterward
def download_files_from_dropbox(dropbox_folder, local_folder, refresh_token, client_id, client_secret, log_file_path):
    access_token = refresh_access_token(refresh_token, client_id, client_secret)
    dbx = dropbox.Dropbox(access_token)

    with open(log_file_path, "a") as log_file:
        log_file.write("🚀 Starting download process (ALL files)...\n")
        try:
            os.makedirs(local_folder, exist_ok=True)

            has_more = True
            cursor = None

            while has_more:
                result = (
                    dbx.files_list_folder_continue(cursor)
                    if cursor else
                    dbx.files_list_folder(dropbox_folder)
                )

                log_file.write(f"📁 Listing files in: {dropbox_folder}\n")

                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        local_path = os.path.join(local_folder, entry.name)

                        # Download file
                        with open(local_path, "wb") as f:
                            _, res = dbx.files_download(path=entry.path_lower)
                            f.write(res.content)

                        log_file.write(f"✅ Downloaded {entry.name} → {local_path}\n")
                        print(f"✅ Downloaded: {entry.name}")

                        # Delete file from Dropbox
                        try:
                            dbx.files_delete_v2(entry.path_lower)
                            log_file.write(f"🗑️ Deleted from Dropbox: {entry.path_lower}\n")
                            print(f"🗑️ Deleted from Dropbox: {entry.name}")
                        except Exception as delete_err:
                            log_file.write(f"❌ Failed to delete {entry.path_lower}: {delete_err}\n")
                            print(f"❌ Failed to delete {entry.name}: {delete_err}")

                has_more = result.has_more
                cursor = result.cursor

            log_file.write("🎉 Download + delete completed.\n")

        except dropbox.exceptions.ApiError as err:
            log_file.write(f"❌ Dropbox API error: {err}\n")
            print(f"❌ Dropbox API error: {err}")

        except Exception as e:
            log_file.write(f"❌ Unexpected error: {e}\n")
            print(f"❌ Unexpected error: {e}")

# Entry point
if __name__ == "__main__":
    dropbox_folder    = sys.argv[1]
    local_folder      = sys.argv[2]
    refresh_token     = sys.argv[3]
    client_id         = sys.argv[4]
    client_secret     = sys.argv[5]
    log_file_path     = sys.argv[6]

    download_files_from_dropbox(
        dropbox_folder,
        local_folder,
        refresh_token,
        client_id,
        client_secret,
        log_file_path
    )



