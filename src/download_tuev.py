import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from tqdm import tqdm

# Constants
BASE_URL = "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/v2.0.1/"
PROGRESS_FILE = "downloaded_files.txt"

def get_links(url, auth):
    """Recursively collect all file links under the given URL."""
    try:
        r = requests.get(url, auth=auth)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to access {url}: {e}")
        return []

    soup = BeautifulSoup(r.text, 'html.parser')
    links = []

    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text()
        if href.startswith("?") or "mailto" in href or "Parent Dir" in text:
            continue
        full_url = urljoin(url, href)
        print(full_url, "full_url")
        if href.endswith('/'):
            # Directory: Recurse
            links.extend(get_links(full_url, auth))
        else:
            # File
            links.append(full_url)

    return links

def read_progress_file(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

def write_progress_file(progress_path, downloaded):
    with open(progress_path, 'a') as f:
        for item in downloaded:
            f.write(item + '\n')

def download_file(url, dest_path, auth):
    try:
        r = requests.get(url, auth=auth, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed: {url} -> {e}")
        return False

def main(username, password, output_dir):
    auth = (username, password)
    progress_path = os.path.join(output_dir, PROGRESS_FILE)
    downloaded_files = read_progress_file(progress_path)

    print("Fetching file list...")
    all_files = get_links(BASE_URL, auth)
    print(f"Total files found: {len(all_files)}")

    new_downloads = []
    for file_url in all_files:
        rel_path = os.path.relpath(urlparse(file_url).path, urlparse(BASE_URL).path)
        local_path = os.path.join(output_dir, rel_path)

        if file_url in downloaded_files:
            print(f"Skipping already downloaded: {rel_path}")
            continue

        print(f"Downloading {rel_path}...")
        if download_file(file_url, local_path, auth):
            new_downloads.append(file_url)

    if new_downloads:
        write_progress_file(progress_path, new_downloads)
        print(f"{len(new_downloads)} new files downloaded and recorded.")
    else:
        print("No new files needed to be downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TUH EEG Event files recursively.")
    parser.add_argument("-u", "--username", required=True, help="Username for authentication")
    parser.add_argument("-p", "--password", required=True, help="Password for authentication")
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    args = parser.parse_args()
    main(args.username, args.password, args.output)
