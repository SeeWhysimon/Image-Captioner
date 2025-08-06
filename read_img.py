import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# ==== 配置路径 ====
coco_root = "./coco_10"
img_dir = os.path.join(coco_root, "train2017")
caption_file = os.path.join(coco_root, "annotations", "captions_train2017.json")

# ==== 加载 caption json ====
with open(caption_file, 'r') as f:
    coco_data = json.load(f)

# ==== 获取第一张图片的信息 ====
first_image = coco_data['images'][0]
image_id = first_image['id']
file_name = first_image['file_name']
img_path = os.path.join(img_dir, file_name)

# ==== 获取对应的 captions（可能有多个）====
captions = [ann['caption'] for ann in coco_data['annotations'] if ann['image_id'] == image_id]

# ==== 输出图片和 captions ====
print(f"Image ID: {image_id}")
print(f"File name: {file_name}")
print("Captions:")
for cap in captions:
    print(f"- {cap}")

# ==== 显示图片 ====
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.title("COCO Image with Captions")
plt.show()
