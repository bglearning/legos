import os
import urllib.request


def fetch_file(file_name, url):
    """Fetch the file from the URL if not present in cache.

    The file is saved at `~/.legos/datasets/`

    Parameters
    ----------
    file_name: str
        Name of the file that should be fetched.
    url: str
        URL to fetch the file from

    Returns
    -------
    Path to the downloaded file.
    """
    cache_dir = os.path.join(os.path.expanduser('~'), '.legos')
    data_dir = os.path.join(cache_dir, 'datasets')

    # Make Directory if necessary
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, file_name)

    # Download not necessary if file already exists
    if os.path.exists(file_path):
        return file_path

    print('Downloading {} from {}'.format(file_name, url))

    urllib.request.urlretrieve(url, file_path)

    print('File downloaded to ', file_path)

    return file_path

