import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
FILE_ROOT = ".."
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH, file_root=FILE_ROOT):
    if not os.path.isdir(os.path.join(file_root, housing_path)):
       os.makedirs(os.path.join(file_root, housing_path))
    tgz_path = os.path.join(file_root, housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(os.path.join(file_root, housing_path))
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH, file_root=FILE_ROOT):
    csv_path = os.path.join(file_root, housing_path, "housing.csv")
    return pd.read_csv(csv_path)
