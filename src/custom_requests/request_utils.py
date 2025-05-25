import os
import shutil
import time
import typing
from urllib.parse import urlparse
import io
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import bs4 as BeautifulSoup # Intentionally using this alias as per original
import requests

from rignak.custom_requests import local_config as config
from rignak.init import ExistingFilename, assert_argument_types

@assert_argument_types
def request_with_retry(
        url: str,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        soup: bool = True,
        headers: dict = config.HEADERS,
        cookies: config.CookieType = config.COOKIES,
        get: typing.Callable = requests.get,
) -> typing.Union[requests.Response, BeautifulSoup.BeautifulSoup, str, None]: # Adjusted return type for soup
    for iteration in range(maximum_retries):
        try:
            if get is requests.get:
                res = get(url, headers=headers, cookies=cookies, verify=True, timeout=config.TIMEOUT)
                res.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                if soup:
                    return BeautifulSoup.BeautifulSoup(res.text, "lxml")
                return res # Return the response object if soup is False
            else: # For custom get functions, assume they handle their own logic
                res = get(url)
                return res # Trust custom function's return
        except requests.exceptions.RequestException as e: # More specific exception handling
            # print(f'request.request_with_retry: Attempt {iteration+1}/{maximum_retries} failed for {url}: {repr(e)}') # Dev logging
            if iteration < maximum_retries - 1:
                time.sleep(config.RETRY_DELAY) # Use a configured delay
            else:
                # print(f'request.request_with_retry: All {maximum_retries} retries failed for {url}. Last error: {repr(e)}') # Dev logging
                return None # Return None after all retries fail
        except Exception as e: # Catch other unexpected errors
            # print(f'request.request_with_retry: An unexpected error occurred for {url}: {repr(e)}') # Dev logging
            return None # Or re-raise, depending on desired error handling policy
    return None


@assert_argument_types
def request_stream(
        url: str,
        payload: dict = {}, # Typically payload is for POST, consider `params` for GET
        headers: dict = config.HEADERS,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        cookies: config.CookieType = config.COOKIES
) -> typing.Optional[io.BytesIO]: # Return type should be BytesIO or similar for raw stream
    for iteration in range(maximum_retries):
        try:
            r = requests.get(url, stream=True, params=payload, headers=headers, cookies=cookies, timeout=config.TIMEOUT)
            r.raise_for_status()
            if r.status_code == 200:
                # To avoid issues with raw stream, read it into BytesIO
                # This is important if the stream is used later and might be closed
                # or if the connection is dropped.
                raw_content = io.BytesIO()
                for chunk in r.iter_content(chunk_size=8192): # Read in chunks
                    raw_content.write(chunk)
                raw_content.seek(0) # Reset stream position to the beginning
                return raw_content
            # Removed 429 handling here as raise_for_status should cover it if it's an error
            # If 429 is a special case needing specific retry logic, it would be more complex.
        except requests.exceptions.RequestException as e:
            # print(f'request.request_stream: Attempt {iteration+1}/{maximum_retries} failed for {url}: {repr(e)}') # Dev logging
            if iteration < maximum_retries - 1:
                time.sleep(config.RETRY_DELAY)
            else:
                # print(f'request.request_stream: All {maximum_retries} retries failed for {url}. Last error: {repr(e)}') # Dev logging
                return None
        except Exception as e:
            # print(f'request.request_stream: An unexpected error occurred for {url}: {repr(e)}') # Dev logging
            return None
    return None


@assert_argument_types
def download_file(
        url: str,
        filename: typing.Optional[str] = None,
        headers: dict = config.HEADERS
) -> typing.Optional[ExistingFilename]:
    if filename is None:
        filename = url_to_filename(url)
    
    file_path = Path(filename)
    # Ensure the directory exists
    os.makedirs(file_path.parent, exist_ok=True)

    if file_path.exists(): # Check if file already exists
        return ExistingFilename(str(file_path))

    raw_stream = request_stream(url, headers=headers)
    if raw_stream is None:
        # print(f"download_file: Failed to get raw stream for {url}") # Dev logging
        return None
        
    try:
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(raw_stream, f)
        return ExistingFilename(str(file_path))
    except Exception as e: # Catch any exception during file writing
        # print(f"download_file: Error writing file {filename} from {url}: {repr(e)}") # Dev logging
        if file_path.exists(): # Attempt to clean up partial file
            try:
                os.remove(file_path)
            except OSError: # as oe_remove:
                # print(f"download_file: Error removing partial file {filename}: {repr(oe_remove)}") # Dev logging
                pass
        return None
    finally:
        if raw_stream and hasattr(raw_stream, 'close'):
            raw_stream.close()


@assert_argument_types
def _convert_image_to_rgb(img: Image.Image) -> Image.Image:
    """Converts a PIL Image object to RGB, handling transparency by pasting on a white background."""
    if img.mode == 'RGB':
        return img

    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ('RGBA', 'LA'):
            alpha_mask = img.split()[-1]
            background.paste(img, (0, 0), alpha_mask)
        elif img.mode == 'P' and 'transparency' in img.info: # Palette with transparency
            img_rgba = img.convert('RGBA') # Convert to RGBA to handle palette transparency
            alpha_mask = img_rgba.split()[3] # Alpha is 4th channel in RGBA
            background.paste(img_rgba, (0, 0), alpha_mask)
        return background
    else: # Other modes (e.g., P without transparency, CMYK, YCbCr)
        return img.convert('RGB')


@assert_argument_types
def download_file_as_jpg(
        url: str,
        filename: typing.Optional[str] = None,
        headers: dict = config.HEADERS
) -> typing.Optional[ExistingFilename]:
    if filename is None:
        filename = url_to_filename(url)
    
    base_filename, _ = os.path.splitext(filename)
    filename_jpg = base_filename + '.jpg'
    file_path = Path(filename_jpg)

    os.makedirs(file_path.parent, exist_ok=True)

    if file_path.exists():
        # print(f"File already exists: {filename_jpg}") # Dev logging
        return ExistingFilename(str(file_path))

    byte_stream = None
    try:
        byte_stream = request_stream(url, headers=headers)
        if byte_stream is None:
            # print(f"download_file_as_jpg: Failed to download image stream from {url}") # Dev logging
            return None

        img = Image.open(byte_stream)
        img = _convert_image_to_rgb(img)
        
        img.save(file_path, 'JPEG', quality=config.JPG_QUALITY, optimize=True) # Use configured quality
        # print(f"Successfully downloaded and converted {url} to {filename_jpg}") # Dev logging
        return ExistingFilename(str(file_path))

    except UnidentifiedImageError:
        # print(f"download_file_as_jpg: Cannot identify image file from {url}.") # Dev logging
        return None
    except Exception as e:
        # print(f"download_file_as_jpg: Error processing/saving {filename_jpg} from {url}: {repr(e)}") # Dev logging
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError: # as oe_remove:
                # print(f"download_file_as_jpg: Error removing partial file {filename_jpg}: {repr(oe_remove)}") # Dev logging
                pass
        return None
    finally:
        if byte_stream and hasattr(byte_stream, 'close'):
            byte_stream.close()


@assert_argument_types
def url_to_filename(url: str) -> str:
    """Converts a URL into a filesystem-friendly filename."""
    parsed_url = urlparse(url)
    filename_parts = [parsed_url.netloc] + [seg for seg in parsed_url.path.split('/') if seg]
    filename = "_".join(filename_parts)
    
    unsafe_chars = [':', '*', '?', '"', '<', '>', '|', ' ', '/', '\\', '&', '=', '#'] # Added '#'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    while "__" in filename: # Normalize multiple underscores
        filename = filename.replace("__", "_")
    filename = filename.strip('_') # Remove leading/trailing underscores
        
    if not filename:
        return "default_filename" # Fallback for empty or invalid URLs
    return filename


@assert_argument_types
def filename_to_url(filename: str) -> str:
    """Attempts to convert a filename (generated by url_to_filename) back into a URL (heuristic)."""
    parts = filename.split('_')
    if not parts:
        return "https://unknown.domain/default_path"

    domain = parts[0]
    path_segments = parts[1:]
    url_path = "/".join(path_segments)
    
    scheme = "https://"
    if domain.lower().startswith("http://") or domain.lower().startswith("https://"):
        return f"{domain}/{url_path}"
    return f"{scheme}{domain}/{url_path}"


if __name__ == '__main__':
    # --- Test url_to_filename and filename_to_url ---
    test_urls_for_naming = [
        "https://i.pximg.net/img-original/img/2023/07/04/02/47/06/109600193_p0.png",
        "http://example.com/some/path with spaces/image.jpeg?query=value&other=param#fragment",
        "https://domain.net/very_long_filename_with_many_segments/and/more/segments/file.name.with.dots.ext"
    ]
    for test_url in test_urls_for_naming:
        print(f"\nOriginal URL: {test_url}")
        generated_fn = url_to_filename(test_url)
        print(f"Generated Filename: {generated_fn}")
        reversed_fn_url = filename_to_url(generated_fn)
        print(f"Reversed URL (heuristic): {reversed_fn_url}")

    # --- Test download_file_as_jpg and download_file ---
    image_download_url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
    
    temp_dir = Path("temp_test_request_utils_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated temporary directory for downloads: {temp_dir.resolve()}")

    # Test JPG download
    jpg_fn_base = os.path.basename(url_to_filename(image_download_url))
    jpg_fn = os.path.splitext(jpg_fn_base)[0] + ".jpg"
    jpg_target_path = temp_dir / jpg_fn
    print(f"\nAttempting JPG download: {image_download_url} -> {jpg_target_path}")
    jpg_dl_result = download_file_as_jpg(image_download_url, filename=str(jpg_target_path))
    if jpg_dl_result:
        print(f"JPG Download successful: {jpg_dl_result}")
    else:
        print("JPG Download failed.")

    # Test original file download
    original_fn = os.path.basename(url_to_filename(image_download_url))
    original_target_path = temp_dir / original_fn
    print(f"\nAttempting original download: {image_download_url} -> {original_target_path}")
    original_dl_result = download_file(image_download_url, filename=str(original_target_path))
    if original_dl_result:
        print(f"Original Download successful: {original_dl_result}")
    else:
        print("Original Download failed.")

    print(f"\nReview files in {temp_dir.resolve()}. Consider manual cleanup or uncomment shutil.rmtree.")
    # To clean up:
    # try:
    #     shutil.rmtree(temp_dir)
    #     print(f"Cleaned up temporary directory: {temp_dir}")
    # except Exception as e:
    #     print(f"Error cleaning up temporary directory {temp_dir}: {e}")
