#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for details.

"""Install the latest wheel published by the PTOAS nightly GitHub release."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_REPOSITORY = "hw-native-sys/PTOAS"
DEFAULT_TAG = "nightly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the latest compatible wheel from a PTOAS GitHub release."
    )
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help=f"GitHub repository (default: {DEFAULT_REPOSITORY})",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"GitHub release tag (default: {DEFAULT_TAG})",
    )
    parser.add_argument(
        "--package",
        default="ptoas",
        help="Distribution to install (default: ptoas)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the wheel without installing it",
    )
    return parser.parse_args()


def github_request(url: str) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ptoas-nightly-wheel-installer",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"GitHub API request failed with HTTP {error.code}: {error.reason}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"unable to reach GitHub API: {error.reason}") from error


def select_wheel(release: object, package: str) -> tuple[str, str]:
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError as error:
        raise RuntimeError(
            "the packaging module is required; install it with 'python -m pip install packaging'"
        ) from error

    if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
        raise RuntimeError("GitHub release response does not contain wheel assets")

    supported_tags = list(sys_tags())
    tag_rank = {tag: rank for rank, tag in enumerate(supported_tags)}
    candidates = []
    for asset in release["assets"]:
        if not isinstance(asset, dict):
            continue
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if not isinstance(name, str) or not name.endswith(".whl") or not isinstance(url, str):
            continue
        try:
            distribution, version, _, wheel_tags = parse_wheel_filename(name)
        except (TypeError, ValueError):
            continue
        if str(distribution) != package.replace("_", "-").lower():
            continue
        matching_ranks = [tag_rank[tag] for tag in wheel_tags if tag in tag_rank]
        if matching_ranks:
            candidates.append((version, min(matching_ranks), name, url))

    if not candidates:
        raise RuntimeError(
            f"no compatible {package} wheel found in the {release.get('tag_name', 'requested')} release"
        )
    _, _, name, url = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
    return name, url


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ptoas-nightly-wheel-installer"},
    )
    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"failed to download wheel: {error}") from error


def main() -> int:
    args = parse_args()
    try:
        release_url = (
            f"https://api.github.com/repos/{args.repository}/releases/tags/{args.tag}"
        )
        release = github_request(release_url)
        wheel_name, wheel_url = select_wheel(release, args.package)
        print(f"Selected wheel: {wheel_name}")
        if args.dry_run:
            print(wheel_url)
            return 0

        with tempfile.TemporaryDirectory(prefix="ptoas-nightly-") as directory:
            wheel_path = Path(directory) / wheel_name
            download(wheel_url, wheel_path)
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                str(wheel_path),
            ]
            subprocess.run(command, check=True)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
