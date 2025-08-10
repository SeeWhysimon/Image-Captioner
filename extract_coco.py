import os
import shutil
import json


def extract_from_coco(coco_root="../coco", output_root=None, percent=0.1):
    subsets = ["train2017", "val2017", "test2017"]

    if output_root is None:
        output_root = f"../coco_{int(percent * 100)}p"

    if not os.path.exists(output_root):
        for subset in subsets:
            image_dir = os.path.join(coco_root, subset)
            ann_file = f"captions_{subset}.json"
            ann_path = os.path.join(coco_root, "annotations", ann_file)

            output_img_dir = os.path.join(output_root, subset)
            output_ann_dir = os.path.join(output_root, "annotations")
            os.makedirs(output_img_dir, exist_ok=True)
            os.makedirs(output_ann_dir, exist_ok=True)

            if not os.path.exists(ann_path):
                print(f"⚠️ Skipping {subset}({ann_file} not found)")
                # 拷贝图像即可
                image_files = sorted(os.listdir(image_dir))
                total = int(len(image_files) * percent)
                for file_name in image_files[:total]:
                    shutil.copy2(os.path.join(image_dir, file_name), os.path.join(output_img_dir, file_name))
                continue

            with open(ann_path, 'r') as f:
                data = json.load(f)

            images = sorted(data["images"], key=lambda x: x["file_name"])
            total = int(len(images) * percent)
            images = images[:total]
            image_ids = set(img["id"] for img in images)

            annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]

            for img in images:
                src = os.path.join(image_dir, img["file_name"])
                dst = os.path.join(output_img_dir, img["file_name"])
                shutil.copy2(src, dst)

            new_data = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "images": images,
                "annotations": annotations
            }

            out_ann_path = os.path.join(output_ann_dir, ann_file)
            with open(out_ann_path, "w") as f:
                json.dump(new_data, f, indent=2)

            print(f"✅ {subset} processed, {len(images)} images extracted.")
    
    print(f"✅ {percent} of coco images extracted.")
    return output_root

if __name__ == "__main__":
    extract_from_coco()