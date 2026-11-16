import os
import subprocess
import sys

from src.config import PROJECT_ROOT


def main():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print(
            "Download kaggle.json from Kaggle -> Settings -> API -> Create New Token "
            "and place it at ~/.kaggle/kaggle.json (chmod 600)."
        )
        sys.exit(1)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "xhlulu/140k-real-and-fake-faces",
            "-p",
            data_dir,
            "--unzip",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
