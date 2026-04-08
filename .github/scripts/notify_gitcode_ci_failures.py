#!/usr/bin/env python3
"""Poll open GitCode PRs and forward new CI failures to Feishu."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def env_text(name: str, default: str | None = None) -> str:
    raw = os.environ.get(name, default)
    if raw is None:
        raise KeyError(name)
    return raw.strip()


GITCODE_OWNER = env_text("GITCODE_OWNER")
GITCODE_REPO = env_text("GITCODE_REPO")
GITCODE_TOKEN = env_text("GITCODE_TOKEN")
FEISHU_BOT_WEBHOOK = env_text("FEISHU_BOT_WEBHOOK")
FEISHU_BOT_SECRET = env_text("FEISHU_BOT_SECRET", "")
ALERT_STATE_FILE = Path(env_text("ALERT_STATE_FILE", ".cache/gitcode-ci-alert-state.json"))


def log(message: str) -> None:
    print(message, flush=True)


def http_json(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, data: dict | None = None):
    req = urllib.request.Request(url, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)

    payload = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, data=payload) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {body}") from exc

    return json.loads(raw) if raw else None


def gitcode_get(path: str, params: dict[str, str | int] | None = None):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return http_json(
        f"https://api.gitcode.com{path}{query}",
        headers={"PRIVATE-TOKEN": GITCODE_TOKEN},
    )


def list_open_prs() -> list[dict]:
    pulls: list[dict] = []
    page = 1
    while True:
        batch = gitcode_get(
            f"/api/v5/repos/{GITCODE_OWNER}/{GITCODE_REPO}/pulls",
            {"state": "open", "per_page": 100, "page": page},
        )
        if not batch:
            return pulls
        pulls.extend(batch)
        page += 1


def normalize_status(raw: str | None) -> str:
    return (raw or "").strip().lower()


def is_failed_status(status: str) -> bool:
    return status in {"failed", "canceled", "cancelled"}


def load_state() -> dict:
    if not ALERT_STATE_FILE.exists():
        return {"alerts": {}}

    try:
        return json.loads(ALERT_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"alerts": {}}


def save_state(state: dict) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def alert_key(pr: dict) -> str:
    pipeline_id = pr.get("head_pipeline_id")
    if pipeline_id:
        return f"{pr['number']}:{pipeline_id}"
    status = normalize_status(pr.get("pipeline_status") or pr.get("pipeline_status_with_code_quality"))
    updated_at = (pr.get("updated_at") or "").strip()
    return f"{pr['number']}:{status}:{updated_at}"


def feishu_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def feishu_signed_payload(text: str) -> dict:
    payload = {
        "msg_type": "text",
        "content": {
            "text": text,
        },
    }
    if not FEISHU_BOT_SECRET:
        return payload

    timestamp = str(int(time.time()))
    string_to_sign = f"{timestamp}\n{FEISHU_BOT_SECRET}".encode("utf-8")
    digest = hmac.new(
        string_to_sign,
        digestmod=hashlib.sha256,
    ).digest()
    payload["timestamp"] = timestamp
    payload["sign"] = base64.b64encode(digest).decode("utf-8")
    return payload


def notify_feishu(message: str) -> None:
    payload = feishu_signed_payload(message)
    http_json(
        FEISHU_BOT_WEBHOOK,
        method="POST",
        headers=feishu_headers(),
        data=payload,
    )


def pr_web_url(pr: dict) -> str:
    url = (pr.get("html_url") or "").strip()
    if url:
        return url
    return f"https://gitcode.com/{GITCODE_OWNER}/{GITCODE_REPO}/merge_requests/{pr['number']}"


def build_message(pr: dict, status: str) -> str:
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    author = (pr.get("author") or {}).get("name") or (pr.get("user") or {}).get("login") or "unknown"
    pipeline_id = pr.get("head_pipeline_id") or "unknown"
    title = pr.get("title") or "(no title)"
    return (
        f"GitCode CI 失败提醒\n"
        f"仓库: {GITCODE_OWNER}/{GITCODE_REPO}\n"
        f"PR: !{pr['number']} {title}\n"
        f"状态: {status}\n"
        f"Pipeline: {pipeline_id}\n"
        f"分支: {head.get('ref', '?')} -> {base.get('ref', '?')}\n"
        f"作者: {author}\n"
        f"更新时间: {pr.get('updated_at', 'unknown')}\n"
        f"链接: {pr_web_url(pr)}"
    )


def main() -> int:
    state = load_state()
    known_alerts = dict(state.get("alerts") or {})
    current_alerts: dict[str, dict] = {}
    notified = 0

    for pr in list_open_prs():
        status = normalize_status(pr.get("pipeline_status") or pr.get("pipeline_status_with_code_quality"))
        if not is_failed_status(status):
            continue

        key = alert_key(pr)
        current_alerts[key] = {
            "pr_number": pr["number"],
            "status": status,
            "pipeline_id": pr.get("head_pipeline_id"),
            "updated_at": pr.get("updated_at"),
        }

        if key in known_alerts:
            continue

        message = build_message(pr, status)
        notify_feishu(message)
        notified += 1
        log(f"Sent Feishu alert for GitCode PR #{pr['number']} (status: {status})")

    state["alerts"] = current_alerts
    save_state(state)
    log(f"Finished polling GitCode open PRs. New alerts sent: {notified}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
