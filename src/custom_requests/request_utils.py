import os
import shutil
import time
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from http.cookiejar import CookieJar
from rignak.src.logging_utils import logger
import bs4
import requests
from typing import Optional, Callable, Any, Union, Dict, BinaryIO

from . import config


def request_with_retry(
        url: str,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        soup: bool = True,
        headers: Dict[str, Any] = config.HEADERS,
        cookies: Optional[CookieJar] = config.COOKIES,
        get: Callable[..., requests.Response] = requests.get,
) -> Union[requests.Response, bs4.BeautifulSoup, int, None]:
    current_response: Union[requests.Response, bs4.BeautifulSoup]
    for iteration in range(maximum_retries):
        response = None
        try:
            # Assuming 'get' is requests.get or a compatible callable
            response = get(url, headers=headers, cookies=cookies, verify=True, timeout=config.THREAD_TIMEOUT)
            if isinstance(response, requests.Response):
                response.raise_for_status()
                response = response.text

            if soup:
                response = bs4.BeautifulSoup(response, "lxml")
            return response
        except requests.exceptions.HTTPError:
            if response.status_code == 429:
                logger(f"Error429 - Waiting {config.ERROR429_COOLDOWN}s")
                time.sleep(config.ERROR429_COOLDOWN)
            elif response.status_code == 111:
                logger(f"Error111 - Early return")
                return 111
            else:
                return response.status_code

        except (requests.exceptions.RequestException, Exception) as e:
            logger(f"{url} (will retry) -> {repr(e)}")
            if iteration < maximum_retries - 1:
                time.sleep(config.SECONDS_BETWEEN_THREADS)
    return None


def request_stream(
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        headers: Dict[str, Any] = config.HEADERS,
        maximum_retries: int = config.MAXIMUM_RETRIES,
        cookies: Optional[CookieJar] = config.COOKIES,
) -> Optional[BinaryIO]:
    response: requests.Response
    for iteration in range(maximum_retries):
        try:
            response = requests.get(
                url,
                stream=True,
                data=payload or {},
                headers=headers,
                cookies=cookies,
                timeout=config.THREAD_TIMEOUT
            )

            response.raise_for_status()
            return response.raw

        except (requests.exceptions.HTTPError, requests.exceptions.RequestException, Exception) as e:
            logger(f"{url} -> {e}")
            if iteration < maximum_retries - 1:
                time.sleep(config.SECONDS_BETWEEN_THREADS)
    return None

def download_from_youtube(url: str, filename: str, lowres:bool=True) -> None:
    import yt_dlp
        # These are the options passed to yt_dlp
    ydl_opts = {'outtmpl': f'{os.path.splitext(filename)[0]}'}
    if lowres:
        ydl_opts['format'] = 'bestvideo[height<=480]+bestaudio[abr<=96]/best[height<=480]'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except yt_dlp.DownloadError as e:
            logger(f"Error downloading {url}: {e}")


def download_file(
        url: str,
        filename: Optional[str] = None,
        headers: Dict[str, Any] = config.HEADERS
) -> Optional[str]:
    effective_filename: Path = Path(url_to_filename(url)) if filename is None else Path(filename)

    if effective_filename.exists():
        return str(effective_filename)

    effective_filename.parent.mkdir(parents=True, exist_ok=True)

    raw_stream: Optional[BinaryIO] = None
    try:
        raw_stream = request_stream(url, headers=headers)
        if raw_stream is None:
            return None  # request_stream failed

        with open(effective_filename, 'wb') as file_handle:
            shutil.copyfileobj(raw_stream, file_handle)
        return str(effective_filename)
    except IOError:  # More specific than AttributeError for file operations
        # Attempt to clean up partially written file if copyfileobj fails
        if effective_filename.exists():
            try:
                os.remove(effective_filename)
            except OSError:  # Ignore error during cleanup
                pass
        return None
    finally:
        if raw_stream and hasattr(raw_stream, 'close') and callable(raw_stream.close):
            try:
                raw_stream.close()
            except Exception:  # Ignore errors on close, e.g. if already closed
                pass


def download_file_as_jpg(
        url: str,
        filename: Optional[str] = None,
        headers: Dict[str, Any] = config.HEADERS
) -> Optional[str]:
    base_name_str: str
    if filename:
        base_name_str = Path(filename).stem
        output_dir = Path(filename).parent
    else:
        base_name_str = Path(url_to_filename(url)).stem
        output_dir = Path(".")

    jpg_file_path: Path = output_dir / f"{base_name_str}.jpg"

    output_dir.mkdir(parents=True, exist_ok=True)

    # if jpg_file_path.exists():
    #     return str(jpg_file_path)

    byte_stream: Optional[BinaryIO] = None
    pil_image: Optional[Image.Image] = None
    try:
        byte_stream = request_stream(url, headers={})
        if byte_stream is None:
            return None

        pil_image = Image.open(byte_stream)  # Image.open can take a file-like object

        # Convert to RGB if necessary (JPG doesn't support transparency/alpha)
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            if pil_image.mode == 'P' and 'transparency' in pil_image.info:
                pil_image = pil_image.convert('RGBA')  # Convert palette with transparency to RGBA first

            if pil_image.mode in ('RGBA', 'LA'):  # Check again if conversion happened or was already RGBA/LA
                # Create a new RGB image with a white background if alpha channel is present
                background = Image.new("RGB", pil_image.size, (255, 255, 255))
                alpha_mask = pil_image.split()[-1]  # Get the alpha channel (last channel for RGBA, second for LA)
                background.paste(pil_image, mask=alpha_mask)
                pil_image.close()  # Close original image after pasting
                pil_image = background
            else:  # Palette without transparency info or other modes
                converted_image = pil_image.convert('RGB')
                if converted_image is not pil_image:  # If convert creates a new image
                    pil_image.close()
                    pil_image = converted_image
        elif pil_image.mode != 'RGB':
            converted_image = pil_image.convert('RGB')
            if converted_image is not pil_image:
                pil_image.close()
                pil_image = converted_image

        jpg_quality = getattr(config, 'JPG_QUALITY', 90)  # Ensure JPG_QUALITY is available in config, else default
        pil_image.save(jpg_file_path, 'JPEG', quality=jpg_quality, optimize=True)
        return str(jpg_file_path)

    except UnidentifiedImageError as e:
        logger(f"UnidentifiedImageError: {url} -> {e}")
        return None
    except Exception as e:
        logger(f"{url} -> {e}")
        # Attempt to clean up partially written file
        if jpg_file_path.exists():
            try:
                os.remove(jpg_file_path)
            except OSError:  # Ignore error during cleanup
                pass
        return None
    finally:
        if byte_stream and hasattr(byte_stream, 'close') and callable(byte_stream.close):
            try:
                byte_stream.close()
            except Exception:
                pass
        if pil_image and hasattr(pil_image, 'close') and callable(pil_image.close):  # Close image if opened
            try:
                pil_image.close()
            except Exception:
                pass


def url_to_filename(url: str) -> str:
    try:
        parsed_url = urlparse(url)
        # Use path component; if path is empty or just '/', use netloc
        if parsed_url.path and Path(parsed_url.path).name:
            filename_candidate = Path(parsed_url.path).name
        else:
            filename_candidate = parsed_url.netloc

        if not filename_candidate:  # Fallback if both path and netloc are problematic
            return "downloaded_file_from_url"

        # Basic sanitization: replace non-alphanumeric (excluding dot, hyphen, underscore) with underscore
        sanitized_filename = "".join(c if c.isalnum() or c in '.-_' else '_' for c in filename_candidate)

        max_len = 200  # Common OS limit is 255, leave some room
        if len(sanitized_filename) > max_len:
            name_part, ext_part = os.path.splitext(sanitized_filename)
            ext_len = len(ext_part)
            name_part = name_part[:max_len - ext_len]
            sanitized_filename = name_part + ext_part

        return sanitized_filename if sanitized_filename else "default_filename"
    except Exception:  # Catch any parsing errors
        return "failed_to_parse_url"


def filename_to_url(filename: str) -> str:
    url_intermediate: str = filename
    url_intermediate = url_intermediate.replace(' ', '/')
    return 'https:' + url_intermediate


if __name__ == '__main__':
    pass
