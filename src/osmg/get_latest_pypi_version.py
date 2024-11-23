"""Determine the latest version of the osmg package on PyPI."""

import argparse

import requests


def get_latest_version(package_name: str) -> str:
    """
    Determine the latest version of the osmg package on PyPI.

    Returns:
      The latest version.
    """
    url = f'https://pypi.org/pypi/{package_name}/json'

    response = requests.get(url, timeout=10)
    package_info = response.json()
    return str(package_info['info']['version'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--package_name', default='osmg', help='Name of the package.'
    )
    args = parser.parse_args()

    package = args.package_name
    version = get_latest_version(package)
    print(version)  # noqa: T201
