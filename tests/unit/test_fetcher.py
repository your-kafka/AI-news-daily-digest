import httpx
import pytest

from ai_news.scrapers import fetcher as fetcher_module
from ai_news.scrapers.fetcher import USER_AGENT, Fetcher, FetchError, describe_failure, retry_delay

ROBOTS = "User-agent: *\nDisallow: /private/\n"


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(fetcher_module.time, "sleep", calls.append)
    return calls


def make_fetcher(handler, **options) -> Fetcher:
    fetcher = Fetcher(min_interval=0, **options)
    fetcher._client.close()
    fetcher._client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT})
    return fetcher


def site(responses: dict[str, list[httpx.Response]], robots: str | None = ROBOTS):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots) if robots is not None else httpx.Response(404)
        queue = responses[request.url.path]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return handler, requests


def test_gets_page_with_user_agent_and_caches_robots(sleeps):
    handler, requests = site({"/page": [httpx.Response(200, text="hello")]})
    with make_fetcher(handler) as fetcher:
        assert fetcher.get("https://example.com/page").text == "hello"
        fetcher.get("https://example.com/page")
    assert [request.url.path for request in requests] == ["/robots.txt", "/page", "/page"]
    assert requests[1].headers["User-Agent"] == USER_AGENT


def test_respects_robots_txt(sleeps):
    handler, requests = site({})
    with make_fetcher(handler) as fetcher, pytest.raises(FetchError, match="robots.txt disallows"):
        fetcher.get("https://example.com/private/page")
    assert [request.url.path for request in requests] == ["/robots.txt"]


def test_missing_robots_txt_allows_everything(sleeps):
    handler, _ = site({"/private/page": [httpx.Response(200, text="ok")]}, robots=None)
    with make_fetcher(handler) as fetcher:
        assert fetcher.get("https://example.com/private/page").text == "ok"


def test_retries_retryable_statuses_honouring_retry_after(sleeps):
    responses = [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(503), httpx.Response(200)]
    handler, _ = site({"/page": responses})
    with make_fetcher(handler) as fetcher:
        assert fetcher.get("https://example.com/page").status_code == 200
    assert sleeps == [7.0, 2.0]


def test_gives_up_after_retries(sleeps):
    handler, _ = site({"/page": [httpx.Response(503)]})
    with make_fetcher(handler, retries=2) as fetcher, pytest.raises(FetchError, match="HTTP 503"):
        fetcher.get("https://example.com/page")
    assert len(sleeps) == 2


def test_client_errors_are_not_retried(sleeps):
    handler, requests = site({"/page": [httpx.Response(404)]})
    with make_fetcher(handler) as fetcher, pytest.raises(FetchError, match="HTTP 404"):
        fetcher.get("https://example.com/page")
    assert len(requests) == 2


def test_network_errors_become_fetch_errors(sleeps):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("refused", request=request)

    with make_fetcher(handler) as fetcher, pytest.raises(FetchError, match="ConnectError"):
        fetcher.get("https://example.com/page")


def test_waits_between_requests_to_the_same_host(sleeps):
    handler, _ = site({"/page": [httpx.Response(200)]})
    fetcher = make_fetcher(handler)
    fetcher.min_interval = 60
    fetcher.get("https://example.com/page")
    assert len(sleeps) == 1
    assert 55 < sleeps[0] <= 60


def test_retry_delay():
    assert retry_delay(httpx.Response(429, headers={"Retry-After": "5"}), 0) == 5
    assert retry_delay(httpx.Response(429, headers={"Retry-After": "999"}), 0) == 60
    assert retry_delay(httpx.Response(503), 3) == 8
    assert retry_delay(httpx.Response(503, headers={"Retry-After": "Wed, 21 Oct 2026"}), 1) == 2


def test_describe_failure_includes_json_bodies_only():
    request = httpx.Request("GET", "https://api.example.com/x")
    json_response = httpx.Response(403, json={"error": "quota"}, request=request)
    html_response = httpx.Response(500, text="<html>oops</html>", request=request)
    assert describe_failure(json_response) == 'HTTP 403 from https://api.example.com/x: {"error":"quota"}'
    assert describe_failure(html_response) == "HTTP 500 from https://api.example.com/x"
