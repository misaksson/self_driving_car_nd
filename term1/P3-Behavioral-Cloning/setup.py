import zipfile
import urllib.request
import os.path


def download_and_unzip(input_url, output_dir):
    if os.path.exists(output_dir):
        print("The output dir {} already exist, skipping download.".format(output_dir))
        return False
    else:
        print("Downloading file {}".format(input_url))
        filehandle, _ = urllib.request.urlretrieve(input_url)

        print("Unzipping file to {}".format(output_dir))
        zipfile.ZipFile(filehandle).extractall(output_dir)
        return True


input_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip"
output_dir = "./simulator/"
download_and_unzip(input_url, output_dir)
