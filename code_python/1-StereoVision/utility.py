def install(package):
    from sys import executable
    from subprocess import check_call
    print(f"Installing {package}...")
    check_call([executable, "-m", "pip", "install", package])


def download_file(url: str, file_name: str):
    from urllib import request
    print(f"Downloading {file_name} from:\n{url}")
    result = request.urlretrieve(url, file_name)
    print("Downloading finished!")
    return result


def install_requirements(requirements: str = "requirements.txt",
                         url: str = None):
    from sys import executable
    from subprocess import check_call
    from os import path
    if path.exists(requirements):
        check_call([executable, "-m", "pip", "install", "-r", requirements])
    elif url is not None:
        download_file(url, requirements)
        check_call([executable, "-m", "pip", "install", "-r", requirements])
    else:
        print(f"Unable to locate {requirements}. Provide a URL to download "
              f"the file if this didn't exists.")
