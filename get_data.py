import os
import zipfile
import requests
from tqdm import tqdm  # 进度条工具

# COCO 2017 数据集下载链接
COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# 下载目录
DOWNLOAD_DIR = "coco"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def unzip_file(zip_path, extract_to):
    """解压文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # 可选：删除压缩包

def download_and_extract_coco():
    """下载并解压所有 COCO 文件"""
    for name, url in COCO_URLS.items():
        zip_path = os.path.join(DOWNLOAD_DIR, f"{name}.zip")
        print(f"Downloading {name}...")
        download_file(url, zip_path)
        print(f"Unzipping {name}...")
        unzip_file(zip_path, DOWNLOAD_DIR)

if __name__ == "__main__":
    download_and_extract_coco()
    print("COCO 数据集下载和解压完成！")