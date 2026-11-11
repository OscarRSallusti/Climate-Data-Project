"""Small JSON client that respects the rate limits of the free data APIs."""

from __future__ import annotations

import time

import requests

_session = requests.Session()
_session.headers["User-Agent"] = "climate-power-research/0.1"


class RateLimitError(RuntimeError):
    pass


def get_json(url: str, params: dict, max_tries: int = 8):
    for attempt in range(max_tries):
        try:
            r = _session.get(url, params=params, timeout=180)
        except requests.RequestException:
            if attempt == max_tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:  # truncated body from an overloaded server
                time.sleep(5 * (attempt + 1))
                continue
        if r.status_code == 429:
            text = r.text.lower()
            if "hourly" in text or "daily" in text:
                # Open-Meteo's long-window quotas: retrying now is pointless.
                # Completed chunks are cached, so a later run resumes.
                raise RateLimitError(f"{url}: {r.text[:200]}")
            wait = float(r.headers.get("retry-after") or 60) + 2
            print(f"  rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"{url} -> HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"{url}: gave up after {max_tries} attempts")
