"""Install the pinned Avalanche source and project overlays in the active env."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "eb075be393e1f458b2c352514ff6c17b5a2c0f4e"


def main():
    checkout = ROOT / ".deps" / "avalanche"
    if not (checkout / ".git").is_dir():
        checkout.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin",
                        "https://github.com/ContinualAI/avalanche.git"], check=True)
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != COMMIT:
        subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", COMMIT], check=True)
    source = ROOT / "modified" / "avalanche"
    target = checkout / "avalanche"
    for path in source.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # Extend the pinned upstream exports instead of overwriting its initializer.
    relative = "avalanche/benchmarks/classic/__init__.py"
    upstream = subprocess.check_output(["git", "-C", str(checkout), "show",
                                        f"{COMMIT}:{relative}"], text=True)
    (checkout / relative).write_text(upstream.rstrip() + "\nfrom .csoftrobot import SoftRobot\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "--no-build-isolation", "-e", str(checkout)], check=True)


if __name__ == "__main__":
    main()
