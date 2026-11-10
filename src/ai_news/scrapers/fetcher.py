import time
from typing import Any, Self
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "Lodestar/0.1 (AI news research project)"
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY_SECONDS = 60.0


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(self, min_interval: float = 1.0, retries: int = 3, timeout: float = 30.0) -> None:
        self.min_interval = min_interval
        self.retries = retries
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            transport=httpx.HTTPTransport(retries=2),
        )
        self._last_request_at: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._client.close()

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        try:
            if not self.allowed(url):
                raise FetchError(f"robots.txt disallows {url}")
            return self._get_with_retries(url, params, headers)
        except httpx.HTTPStatusError as error:
            raise FetchError(describe_failure(error.response)) from error
        except httpx.HTTPError as error:
            raise FetchError(f"{type(error).__name__} while fetching {url}") from error

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        return self._robots[origin].can_fetch(USER_AGENT, url)

    def _load_robots(self, origin: str) -> RobotFileParser:
        robots = RobotFileParser()
        self._wait_for_turn(origin)
        response = self._client.get(f"{origin}/robots.txt")
        if response.status_code == 200:
            robots.parse(response.text.splitlines())
        else:
            robots.allow_all = True
        return robots

    def _get_with_retries(
        self,
        url: str,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        attempt = 0
        while True:
            self._wait_for_turn(url)
            response = self._client.get(url, params=params, headers=headers)
            if response.status_code in RETRYABLE_STATUSES and attempt < self.retries:
                time.sleep(retry_delay(response, attempt))
                attempt += 1
                continue
            response.raise_for_status()
            return response

    def _wait_for_turn(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self._last_request_at.get(host, float("-inf")) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_at[host] = time.monotonic()


def retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After", "")
    delay = float(retry_after) if retry_after.isdigit() else 2.0**attempt
    return min(delay, MAX_RETRY_DELAY_SECONDS)


def describe_failure(response: httpx.Response) -> str:
    description = f"HTTP {response.status_code} from {response.request.url}"
    if response.headers.get("Content-Type", "").startswith("application/json"):
        description += f": {response.text[:300]}"
    return description
