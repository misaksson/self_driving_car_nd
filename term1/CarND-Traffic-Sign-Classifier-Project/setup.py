
import zipfile
import urllib.request
import os.path

input_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
output_dir = "./data/"

if os.path.exists(output_dir):
    print("The output dir {} already exist, skipping download.".format(output_dir))
else:
    print("Downloading file {}".format(input_url))
    filehandle, _ = urllib.request.urlretrieve(input_url)

    print("Unzipping file to {}".format(output_dir))
    zipfile.ZipFile(filehandle).extractall(output_dir)
