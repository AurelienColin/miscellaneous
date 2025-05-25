from http.cookiejar import CookieJar

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Cache-Control': 'no-cache',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.pixiv.net/'
}
MAXIMUM_RETRIES = 5

CookieType = CookieJar
try:
    import browsercookie

    COOKIES = browsercookie.load()

except (UnicodeDecodeError, ImportError) as e:
    COOKIES = None
    print(f"Error when reading Cookies: {e}")

PARALLEL_THREADS_LIMIT = 20
SECONDS_BETWEEN_THREADS = 0
THREAD_TIMEOUT = 10
SECONDS_AFTER_COMPLETION = 10
JPG_QUALITY = 95