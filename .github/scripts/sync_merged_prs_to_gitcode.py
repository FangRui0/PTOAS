#!/usr/bin/env python3
"""Poll merged upstream PRs and mirror unsynced ones to GitCode in batches."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def env_text(name: str, default: str | None = None) -> str:
    raw = os.environ.get(name, default)
    if raw is None:
        raise KeyError(name)
    return raw.strip()


UPSTREAM_REPO = env_text("UPSTREAM_REPO")
UPSTREAM_OWNER, UPSTREAM_NAME = UPSTREAM_REPO.split("/", 1)

GITCODE_OWNER = env_text("GITCODE_OWNER")
GITCODE_REPO = env_text("GITCODE_REPO")
GITCODE_REPO_SSH = env_text("GITCODE_REPO_SSH")
GITCODE_TOKEN = env_text("GITCODE_TOKEN")

GH_TOKEN = env_text("GH_TOKEN", "")
INPUT_PR_NUMBER = env_text("INPUT_PR_NUMBER", "")
LOOKBACK_DAYS = int(env_text("LOOKBACK_DAYS", "3"))
SYNC_BASELINE_PR_NUMBER = int(env_text("SYNC_BASELINE_PR_NUMBER", "340") or "340")
FORCE_RESYNC = env_text("FORCE_RESYNC", "false").lower() == "true"
GITCODE_COMMIT_NAME = env_text("GITCODE_COMMIT_NAME", "FangRui")
GITCODE_COMMIT_EMAIL = env_text("GITCODE_COMMIT_EMAIL", "fangrui0827@gmail.com")
GITCODE_AUTHOR = f"{GITCODE_COMMIT_NAME} <{GITCODE_COMMIT_EMAIL}>"

BATCH_MARKER_PREFIX = "github-pr-sync-batch"
ITEM_MARKER_PREFIX = "github-pr-sync-item"


@dataclass(frozen=True)
class OpenBatchPR:
    number: int
    head_ref: str
    included_numbers: tuple[int, ...]


@dataclass(frozen=True)
class BatchTarget:
    gitcode_base_ref: str
    github_prs: list[dict]
    prs_to_apply: list[dict]
    open_gitcode_pr: OpenBatchPR | None


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], cwd: str | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


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


def gh_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
    return headers


def gh_get(path: str, params: dict[str, str | int] | None = None):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return http_json(f"https://api.github.com{path}{query}", headers=gh_headers())


def gitcode_headers() -> dict[str, str]:
    return {"PRIVATE-TOKEN": GITCODE_TOKEN}


def gitcode_get(path: str, params: dict[str, str | int] | None = None):
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    return http_json(f"https://api.gitcode.com{path}{query}", headers=gitcode_headers())


def gitcode_write(path: str, method: str, data: dict):
    return http_json(
        f"https://api.gitcode.com{path}",
        method=method,
        headers=gitcode_headers(),
        data=data,
    )


def gitcode_comment_on_pr(pr_number: int, body: str) -> None:
    gitcode_write(
        f"/api/v5/repos/{GITCODE_OWNER}/{GITCODE_REPO}/pulls/{pr_number}/comments",
        "POST",
        {
            "body": body,
            "need_to_resolve": False,
        },
    )


def parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def marker_for_batch(base_ref: str) -> str:
    return f"<!-- {BATCH_MARKER_PREFIX}:{UPSTREAM_REPO}:{base_ref} -->"


def marker_for_pr(pr_number: int) -> str:
    return f"<!-- {ITEM_MARKER_PREFIX}:{UPSTREAM_REPO}#{pr_number} -->"


def batch_branch_for(base_ref: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", base_ref).strip("-")
    return f"sync/github-merged-prs-{sanitized or 'default'}"


def git_ref_exists(repo_dir: str, ref: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", ref],
        cwd=repo_dir,
        check=False,
    )
    return result.returncode == 0


def remote_head_branch(repo_dir: str) -> str | None:
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
        cwd=repo_dir,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    ref = result.stdout.strip()
    prefix = "refs/remotes/origin/"
    if ref.startswith(prefix):
        return ref[len(prefix):]
    return None


def resolve_gitcode_base_branch(repo_dir: str, requested_ref: str) -> str:
    candidates = [requested_ref]
    if requested_ref == "main":
        candidates.append("master")
    elif requested_ref == "master":
        candidates.append("main")

    head_branch = remote_head_branch(repo_dir)
    if head_branch:
        candidates.append(head_branch)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        if git_ref_exists(repo_dir, f"refs/remotes/origin/{candidate}"):
            return candidate

        fetched = subprocess.run(
            ["git", "fetch", "origin", f"+refs/heads/{candidate}:refs/remotes/origin/{candidate}"],
            cwd=repo_dir,
            check=False,
            text=True,
            capture_output=True,
        )
        if fetched.returncode == 0:
            return candidate

    raise RuntimeError(
        f"Unable to find a matching GitCode base branch for upstream base '{requested_ref}'"
    )


def merge_commit_sha(pr: dict) -> str:
    sha = (pr.get("merge_commit_sha") or "").strip()
    if not sha:
        raise RuntimeError(f"GitHub PR #{pr['number']} is missing merge_commit_sha")
    return sha


def commit_parent_count(repo_dir: str, commit: str) -> int:
    result = run(
        ["git", "rev-list", "--parents", "-n", "1", commit],
        cwd=repo_dir,
        capture=True,
    )
    parts = result.stdout.strip().split()
    if not parts:
        raise RuntimeError(f"Unable to inspect commit parents for {commit}")
    return max(0, len(parts) - 1)


def is_gitcode_pr_merged(pr: dict) -> bool:
    return bool(pr.get("merged_at")) or pr.get("state") == "merged"


def extract_sync_numbers(pr: dict) -> set[int]:
    body = pr.get("body") or ""
    numbers = {
        int(match.group(2))
        for match in re.finditer(r"<!-- github-pr-sync-item:([^#]+/[^#]+)#(\d+) -->", body)
        if match.group(1) == UPSTREAM_REPO
    }

    legacy = re.search(r"<!-- github-pr-sync:([^#]+/[^#]+)#(\d+) -->", body)
    if legacy and legacy.group(1) == UPSTREAM_REPO:
        numbers.add(int(legacy.group(2)))

    return numbers


def is_batch_sync_pr(pr: dict) -> bool:
    body = pr.get("body") or ""
    return marker_for_batch(pr["base"]["ref"]) in body or bool(extract_sync_numbers(pr))


def list_recent_merged_prs() -> list[dict]:
    if INPUT_PR_NUMBER:
        pr = gh_get(f"/repos/{UPSTREAM_OWNER}/{UPSTREAM_NAME}/pulls/{INPUT_PR_NUMBER}")
        if not pr.get("merged_at"):
            raise RuntimeError(f"GitHub PR #{INPUT_PR_NUMBER} is not merged")
        return [pr]

    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    merged_prs: list[dict] = []
    page = 1

    while True:
        batch = gh_get(
            f"/repos/{UPSTREAM_OWNER}/{UPSTREAM_NAME}/pulls",
            {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )
        if not batch:
            break

        stop = False
        for pr in batch:
            updated_at = parse_timestamp(pr["updated_at"])
            if updated_at < cutoff:
                stop = True
                break
            if pr.get("merged_at"):
                merged_prs.append(pr)

        if stop:
            break
        page += 1

    merged_prs.sort(key=lambda item: (parse_timestamp(item["merged_at"]), item["number"]))
    return merged_prs


def load_github_pr(pr_number: int) -> dict:
    pr = gh_get(f"/repos/{UPSTREAM_OWNER}/{UPSTREAM_NAME}/pulls/{pr_number}")
    if not pr.get("merged_at"):
        raise RuntimeError(f"GitHub PR #{pr_number} is not merged")
    return pr


def baseline_merged_at() -> datetime | None:
    if SYNC_BASELINE_PR_NUMBER <= 0:
        return None
    baseline_pr = load_github_pr(SYNC_BASELINE_PR_NUMBER)
    return parse_timestamp(baseline_pr["merged_at"])


def collect_gitcode_sync_state() -> tuple[set[int], dict[str, OpenBatchPR]]:
    synced_numbers: set[int] = set()
    open_pr_by_base: dict[str, OpenBatchPR] = {}
    page = 1

    while True:
        batch = gitcode_get(
            f"/api/v5/repos/{GITCODE_OWNER}/{GITCODE_REPO}/pulls",
            {
                "state": "all",
                "per_page": 100,
                "page": page,
            },
        )
        if not batch:
            break

        for pr in batch:
            if not is_batch_sync_pr(pr):
                continue

            numbers = extract_sync_numbers(pr)
            if not numbers:
                continue

            if is_gitcode_pr_merged(pr):
                synced_numbers.update(numbers)
                continue

            if pr.get("state") == "open":
                base_ref = pr["base"]["ref"]
                head_ref = ((pr.get("head") or {}).get("ref") or "").strip()
                if not head_ref:
                    head_ref = batch_branch_for(base_ref)
                current = open_pr_by_base.get(base_ref)
                candidate = OpenBatchPR(
                    number=pr["number"],
                    head_ref=head_ref,
                    included_numbers=tuple(sorted(numbers)),
                )
                if current is None or pr["number"] > current.number:
                    open_pr_by_base[base_ref] = candidate

        page += 1

    return synced_numbers, open_pr_by_base


def build_batch_targets() -> list[BatchTarget]:
    recent_prs = {pr["number"]: pr for pr in list_recent_merged_prs()}
    synced_numbers, open_pr_by_base = collect_gitcode_sync_state()
    open_numbers: set[int] = {
        number
        for open_pr in open_pr_by_base.values()
        for number in open_pr.included_numbers
    }

    candidate_numbers = set(recent_prs)
    if not FORCE_RESYNC:
        candidate_numbers.difference_update(synced_numbers)
        candidate_numbers.difference_update(open_numbers)

    if INPUT_PR_NUMBER:
        candidate_numbers = {int(INPUT_PR_NUMBER)}

    github_prs: dict[int, dict] = dict(recent_prs)
    missing_numbers = sorted(candidate_numbers.union(open_numbers).difference(github_prs))
    for pr_number in missing_numbers:
        github_prs[pr_number] = load_github_pr(pr_number)

    baseline_time = None if INPUT_PR_NUMBER else baseline_merged_at()
    selected = [github_prs[number] for number in sorted(candidate_numbers)]
    selected = [pr for pr in selected if pr.get("merged_at")]
    if baseline_time is not None:
        selected = [
            pr
            for pr in selected
            if parse_timestamp(pr["merged_at"]) > baseline_time
        ]
    selected.sort(key=lambda item: (parse_timestamp(item["merged_at"]), item["number"]))

    grouped: dict[str, list[dict]] = {}
    for pr in selected:
        source_base = pr["base"]["ref"]
        gitcode_base = "master" if source_base == "main" else source_base
        grouped.setdefault(gitcode_base, []).append(pr)

    targets: list[BatchTarget] = []
    all_grouped: dict[str, list[dict]] = {}
    for gitcode_base, open_pr in open_pr_by_base.items():
        for pr_number in open_pr.included_numbers:
            pr = github_prs.get(pr_number)
            if pr is not None and (baseline_time is None or parse_timestamp(pr["merged_at"]) > baseline_time):
                all_grouped.setdefault(gitcode_base, []).append(pr)
    for gitcode_base, prs in grouped.items():
        all_grouped.setdefault(gitcode_base, []).extend(prs)

    for gitcode_base, prs in all_grouped.items():
        deduped: dict[int, dict] = {pr["number"]: pr for pr in prs}
        all_prs = sorted(
            deduped.values(),
            key=lambda item: (parse_timestamp(item["merged_at"]), item["number"]),
        )
        open_pr = open_pr_by_base.get(gitcode_base)
        open_numbers_for_base = set(open_pr.included_numbers) if open_pr and not FORCE_RESYNC else set()
        prs_to_apply = [pr for pr in all_prs if pr["number"] not in open_numbers_for_base]
        if not all_prs or (open_pr and not prs_to_apply and not FORCE_RESYNC):
            continue
        targets.append(
            BatchTarget(
                gitcode_base_ref=gitcode_base,
                github_prs=all_prs,
                prs_to_apply=prs_to_apply,
                open_gitcode_pr=open_pr,
            )
        )
    return targets


def commit_one_pr(repo_dir: str, pr: dict) -> bool:
    merged_sha = merge_commit_sha(pr)
    run(["git", "fetch", "upstream", merged_sha], cwd=repo_dir)

    parent_count = commit_parent_count(repo_dir, "FETCH_HEAD")
    cherry_pick_cmd = ["git", "cherry-pick", "--no-commit"]
    if parent_count > 1:
        cherry_pick_cmd.extend(["-m", "1"])
    cherry_pick_cmd.append("FETCH_HEAD")

    cherry_pick = subprocess.run(
        cherry_pick_cmd,
        cwd=repo_dir,
        check=False,
        text=True,
        capture_output=True,
    )
    if cherry_pick.returncode != 0:
        subprocess.run(
            ["git", "cherry-pick", "--abort"],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        details = (cherry_pick.stderr or cherry_pick.stdout or "").strip()
        if not details:
            details = "cherry-pick failed with conflicts"
        raise RuntimeError(
            f"Unable to apply merged commit {merged_sha} for GitHub PR #{pr['number']}: {details}"
        )

    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir)
    if staged.returncode == 0:
        log(f"PR #{pr['number']}: merged commit produces no net changes on GitCode base, skipping")
        return False

    commit_message = (
        f"[GH PR #{pr['number']}] {pr['title']}\n\n"
        f"Source: {pr['html_url']}\n"
        f"Merged at: {pr['merged_at']}\n"
        f"Merge commit: {merged_sha}\n"
    )
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": GITCODE_COMMIT_NAME,
            "GIT_AUTHOR_EMAIL": GITCODE_COMMIT_EMAIL,
            "GIT_COMMITTER_NAME": GITCODE_COMMIT_NAME,
            "GIT_COMMITTER_EMAIL": GITCODE_COMMIT_EMAIL,
        }
    )
    log("+ git commit -m <message>")
    subprocess.run(
        ["git", "commit", f"--author={GITCODE_AUTHOR}", "-m", commit_message],
        cwd=repo_dir,
        check=True,
        text=True,
        env=commit_env,
    )
    return True


def apply_batch_to_gitcode(batch: BatchTarget) -> tuple[bool, str, list[dict]]:
    applied_prs: list[dict] = list(batch.github_prs)

    with tempfile.TemporaryDirectory() as tempdir:
        run(["git", "clone", GITCODE_REPO_SSH, tempdir])
        run(["git", "config", "user.name", GITCODE_COMMIT_NAME], cwd=tempdir)
        run(
            ["git", "config", "user.email", GITCODE_COMMIT_EMAIL],
            cwd=tempdir,
        )
        run(["git", "remote", "add", "upstream", f"https://github.com/{UPSTREAM_REPO}.git"], cwd=tempdir)

        gitcode_base_ref = resolve_gitcode_base_branch(tempdir, batch.gitcode_base_ref)
        local_branch = batch_branch_for(gitcode_base_ref)
        push_ref = batch.open_gitcode_pr.head_ref if batch.open_gitcode_pr and not FORCE_RESYNC else batch_branch_for(gitcode_base_ref)
        if batch.open_gitcode_pr and not FORCE_RESYNC:
            run(
                ["git", "fetch", "origin", f"+refs/heads/{push_ref}:refs/remotes/origin/{push_ref}"],
                cwd=tempdir,
            )
            run(
                ["git", "checkout", "-B", local_branch, f"refs/remotes/origin/{push_ref}"],
                cwd=tempdir,
            )
        else:
            run(
                ["git", "checkout", "-B", local_branch, f"refs/remotes/origin/{gitcode_base_ref}"],
                cwd=tempdir,
            )

        newly_applied: list[dict] = []
        for pr in batch.prs_to_apply:
            log(f"Applying merged GitHub PR #{pr['number']}: {pr['title']}")
            if commit_one_pr(tempdir, pr):
                newly_applied.append(pr)

        if not newly_applied and not batch.open_gitcode_pr:
            return False, gitcode_base_ref, []

        run(
            ["git", "push", "--force", "origin", f"HEAD:refs/heads/{push_ref}"],
            cwd=tempdir,
        )

    return True, gitcode_base_ref, applied_prs


def build_batch_title(prs: list[dict]) -> str:
    first = prs[0]["number"]
    last = prs[-1]["number"]
    if first == last:
        return f"[Batch Sync] GitHub PR #{first}"
    return f"[Batch Sync] GitHub PRs #{first}-#{last} ({len(prs)} items)"


def build_batch_body(prs: list[dict], gitcode_base_ref: str) -> str:
    lines = [
        f"Batch sync from upstream GitHub repo `{UPSTREAM_REPO}`.",
        "",
        f"- GitCode base branch: `{gitcode_base_ref}`",
        f"- Upstream PR count: `{len(prs)}`",
        "",
        "Included merged GitHub PRs:",
    ]
    for pr in prs:
        lines.append(f"- #{pr['number']}: {pr['title']} ({pr['html_url']})")

    lines.extend(
        [
            "",
            marker_for_batch(gitcode_base_ref),
        ]
    )
    for pr in prs:
        lines.append(marker_for_pr(pr["number"]))

    return "\n".join(lines).strip() + "\n"


def upsert_gitcode_batch_pr(
    batch: BatchTarget,
    gitcode_base_ref: str,
    applied_prs: list[dict],
) -> tuple[int, bool]:
    payload = {
        "title": build_batch_title(applied_prs),
        "body": build_batch_body(applied_prs, gitcode_base_ref),
        "base": gitcode_base_ref,
    }

    if batch.open_gitcode_pr is not None:
        gitcode_write(
            f"/api/v5/repos/{GITCODE_OWNER}/{GITCODE_REPO}/pulls/{batch.open_gitcode_pr.number}",
            "PATCH",
            payload,
        )
        return batch.open_gitcode_pr.number, False

    payload["head"] = batch_branch_for(gitcode_base_ref)
    created = gitcode_write(
        f"/api/v5/repos/{GITCODE_OWNER}/{GITCODE_REPO}/pulls",
        "POST",
        payload,
    )
    return created["number"], True


def main() -> int:
    targets = build_batch_targets()
    if not targets:
        log("No merged upstream PRs need syncing.")
        return 0

    failures: list[str] = []
    for batch in targets:
        try:
            applied, resolved_base_ref, applied_prs = apply_batch_to_gitcode(batch)
            if not applied:
                log(f"Base {batch.gitcode_base_ref}: no net changes to sync.")
                continue

            gitcode_pr_number, created_new = upsert_gitcode_batch_pr(
                batch,
                resolved_base_ref,
                applied_prs,
            )
            gitcode_comment_on_pr(gitcode_pr_number, "/compile")
            pr_numbers = ", ".join(f"#{pr['number']}" for pr in applied_prs)
            action = "created" if created_new else "updated"
            log(
                f"Synced batch [{pr_numbers}] to GitCode PR #{gitcode_pr_number} "
                f"({action}, commented /compile)"
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"Base {batch.gitcode_base_ref}: {exc}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
