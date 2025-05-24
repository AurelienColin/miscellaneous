import os
import shutil
import time
import os
import shutil
import time
import typing
from urllib.parse import urlparse
import io  # Added for byte stream handling
from pathlib import Path # Added for cleaner path manipulation
from PIL import Image, UnidentifiedImageError # Added for image processing


import bs4 as BeautifulSoup
import requests

from Rignak.custom_requests import local_config as config
from Rignak.init import ExistingFilename, assert_argument_types


@assert_argument_types
def request_with_retry(
        url: str,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        soup: bool = True,
        headers: dict = config.HEADERS,
        cookies: config.CookieType = config.COOKIES,
        get: typing.Callable = requests.get,
) -> typing.Union[requests.Response, str, None]:
    for iteration in range(maximum_retries):
        try:
            res = get(url, headers=headers, cookies=cookies, verify=True)
            if soup:
                res = BeautifulSoup.BeautifulSoup(res.text, "lxml")
            return res
        except Exception as e:
            print(f'request.request_with_retry: Error: {url} because of "{repr(e)}"')
            time.sleep(10)
    return None


@assert_argument_types
def request_stream(
        url: str,
        payload: dict = {},
        headers: dict = config.HEADERS,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        cookies: config.CookieType = config.COOKIES
) -> None:
    for iteration in range(maximum_retries):
        try:
            r = requests.get(url, stream=True, data=payload, headers=headers, cookies=cookies)
            if r.status_code == 200:
                return r.raw
            elif r.status_code == 429:  # spam
                time.sleep(1)
        except Exception as e:
            print(f'request.request_stream: Error: {url} because of "{repr(e)}"')
    return None


@assert_argument_types
def download_file(
        url: str,
        filename: (str, None) = None,
        headers: dict = config.HEADERS
) -> ExistingFilename:
    if filename is None:
        filename = url_to_filename(url)

    if os.path.exists(filename):
        return filename
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    raw = request_stream(url, headers=headers)
    try:
        with open(filename, 'wb') as file:
            shutil.copyfileobj(raw, file)
        return ExistingFilename(filename)
    except AttributeError:
        os.remove(filename)

@assert_argument_types
def download_file_as_jpg(
        url: str,
        filename: (str, None) = None,
        headers: dict = config.HEADERS
) ->ExistingFilename: # Return ExistingFilename or None on failure
    """
    Downloads a file from a URL, converts it to JPG format, and saves it.
    Assumes the URL points to a valid image file recognizable by Pillow.
    """
    filename = os.path.splitext(filename)[0] + '.jpg'
    file_path = Path(filename)
    os.makedirs(file_path.parent, exist_ok=True)

    if file_path.exists():
        print(f"File already exists: {filename}")
        return ExistingFilename(str(file_path))

    byte_stream = request_stream(url, headers=headers)
    if byte_stream is None:
        return None

    try:
        # Open the image from the byte stream
        img = Image.open(byte_stream)

        # Convert to RGB if necessary (JPG doesn't support transparency/alpha)
        if img.mode in ('RGBA', 'LA', 'P'): # P (Palette) might have transparency
             # Ensure transparent background becomes white (or specify another color)
             # Create a new RGB image with a white background
             background = Image.new("RGB", img.size, (255, 255, 255))
             # Paste the image onto the background using the alpha channel as mask
             # Handle Palette ('P') mode specifically if it has transparency info
             if img.mode == 'P' and 'transparency' in img.info:
                 # Convert palette image to RGBA first to handle transparency correctly
                 img = img.convert('RGBA')
                 background.paste(img, mask=img.split()[3]) # Use alpha channel as mask
                 img = background # The result is the background with the image pasted on it
             elif img.mode in ('RGBA', 'LA'):
                  # For RGBA or LA, use the alpha channel directly
                  alpha_channel = img.split()[-1] # Get the alpha channel
                  background.paste(img, mask=alpha_channel)
                  img = background
             else: # If P mode has no transparency, just convert normally
                  img = img.convert('RGB')
        elif img.mode != 'RGB':
             img = img.convert('RGB')

        img.save(file_path, 'JPEG', quality=95, optimize=True)
        print(f"Successfully downloaded and converted {url} to {filename}")
        return ExistingFilename(str(file_path))

    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file from {url}. It might not be a valid image.")
        return None
    except Exception as e:
        print(f"Error processing or saving image {filename} from {url}: {repr(e)}")
        try:
            if file_path.exists():
                os.remove(file_path)
        except OSError as oe:
            print(f"Error removing partially written file {filename}: {repr(oe)}")
        return None
    finally:
        if byte_stream:
            byte_stream.close()


@assert_argument_types
def url_to_filename(url: str) -> str:
    filename = url.split(':')[1]

    while '/' in filename:
        filename = filename.replace('/', ' ')
    return filename


@assert_argument_types
def filename_to_url(filename: str) -> str:
    url = filename
    while ' ' in url:
        url = url.replace(' ', '/')
    return 'https:' + url


if __name__ == '__main__':
    download_file("https://i.pximg.net/img-original/img/2023/07/04/02/47/06/109600193_p0.png", "1.png")
