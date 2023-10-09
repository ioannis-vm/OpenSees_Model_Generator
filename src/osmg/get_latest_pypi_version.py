import requests


def get_latest_version(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        response = requests.get(url)
        package_info = response.json()
        latest_version = package_info["info"]["version"]
        return latest_version
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    package = "osmg"
    version = get_latest_version(package)
    print(version)
