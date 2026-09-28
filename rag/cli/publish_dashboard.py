from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from rag.experiment_analysis import build_experiment_report
from rag.paths import DEFAULT_EXPERIMENT_REPORT_DIR, DEFAULT_EXPERIMENT_SESSION_ROOT


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild and publish the experiment dashboard to GitHub Pages.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and check the update without publishing.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / DEFAULT_EXPERIMENT_REPORT_DIR
    sessions = root / DEFAULT_EXPERIMENT_SESSION_ROOT
    report = build_experiment_report(sessions, output)
    if not report["records"]:
        raise RuntimeError("No experiment sessions found. Refusing to publish an empty dashboard.")

    remote = git(root, "remote", "get-url", "origin")
    staging = root / "local" / "dashboard_publish"
    staging.mkdir(parents=True, exist_ok=True)
    checkout = Path(tempfile.mkdtemp(prefix="publish_", dir=staging)).resolve()
    print(f"Publication checkout: {checkout}", flush=True)
    git(root, "clone", "--single-branch", "--branch", "gh-pages", remote, str(checkout))
    destination = checkout / DEFAULT_EXPERIMENT_REPORT_DIR
    shutil.copytree(output, destination, dirs_exist_ok=True)
    for record in report["records"]:
        if not record["has_image"]:
            continue
        source = Path(record["image_path"]).resolve()
        source.relative_to(sessions.resolve())
        target = checkout / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    git(checkout, "add", "artifacts/reports/experiment_dashboard", "local/sessions")
    git(checkout, "diff", "--cached", "--check", "--", "*.html", "*.css", "*.js")
    print(git(checkout, "diff", "--cached", "--stat"))
    print(f"Primary conditions: {report['totals']['primary_conditions']}")
    print(f"Missing conditions: {report['totals']['missing_conditions']}")
    if args.dry_run:
        print("Preview complete. Nothing was published.")
        return 0

    if not git(checkout, "diff", "--cached", "--name-only"):
        print("The published dashboard is already current.")
        return 0
    for setting, fallback in (("user.name", "%an"), ("user.email", "%ae")):
        try:
            identity = git(root, "config", "--get", setting)
        except RuntimeError:
            identity = git(root, "log", "-1", f"--format={fallback}")
        git(checkout, "config", setting, identity)
    git(checkout, "commit", "-m", "Update experiment dashboard data")
    git(checkout, "push", "origin", "HEAD:gh-pages")
    print(f"Published commit: {git(checkout, 'rev-parse', 'HEAD')}")
    print("GitHub Pages will deploy the update shortly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
