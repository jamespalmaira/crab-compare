"""Polite HTTP client: per-host rate limit, backoff on errors, descriptive User-Agent."""
import logging
import os
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class MissingContact(RuntimeError):
    pass


def contact_email(env_var):
    """The contact email from the environment; raises if it is unset or empty."""
    email = os.environ.get(env_var, "").strip()
    if not email:
        raise MissingContact(f"Set the {env_var} environment variable to a contact email before using the network.")
    return email


def user_agent(cfg):
    return f"{cfg['user_agent']} contact: {contact_email(cfg['contact_email_env'])}"


class Client:
    def __init__(self, agent, min_intervals=None, default_interval=1.0, timeout=30, max_retries=6,
                 backoff_base=2.0, session=None, sleep=time.sleep, clock=time.monotonic):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = agent
        self.min_intervals = dict(min_intervals or {})
        self.default_interval = default_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.sleep = sleep
        self.clock = clock
        self._last = {}

    @classmethod
    def from_config(cls, cfg, agent=None):
        intervals = {urlparse(cfg["inaturalist"]["base_url"]).netloc: cfg["inaturalist"]["min_interval"],
                     urlparse(cfg["gbif"]["base_url"]).netloc: cfg["gbif"]["min_interval"]}
        h = cfg["http"]
        return cls(agent or user_agent(cfg), intervals, cfg["download"]["min_interval"], h["timeout"],
                      h["max_retries"], h["backoff_base"])

    def _wait(self, host):
        interval = self.min_intervals.get(host, self.default_interval)
        last = self._last.get(host)
        if last is not None:
            gap = interval - (self.clock() - last)
            if gap > 0:
                self.sleep(gap)
        self._last[host] = self.clock()

    def get(self, url, params=None):
        host = urlparse(url).netloc
        for attempt in range(self.max_retries):
            self._wait(host)
            delay = self.backoff_base * 2 ** attempt
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                log.warning("%s: %s (attempt %d/%d)", url, e, attempt + 1, self.max_retries)
            else:
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                    return resp
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                log.warning("%s: HTTP %d (attempt %d/%d)", url, resp.status_code, attempt + 1, self.max_retries)
            if attempt + 1 < self.max_retries:
                self.sleep(delay)
        raise RuntimeError(f"giving up on {url} after {self.max_retries} attempts")

    def get_json(self, url, params=None):
        return self.get(url, params).json()
