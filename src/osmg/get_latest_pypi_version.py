"""
Determine the latest version of the osmg package on PyPI
"""

import argparse
import requests


def get_latest_version(package_name):
    """
    Determine the latest version of the osmg package on PyPI
    """
    url = f"https://pypi.org/pypi/{package_name}/json"

    response = requests.get(url, timeout=10)
    package_info = response.json()
    latest_version = package_info["info"]["version"]
    return latest_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_name", default="osmg", help="Name of the package."
    )
    args = parser.parse_args()

    package = args.package_name
    version = get_latest_version(package)
    print(version)
