# download images from LS
from pathlib import Path
import json
import requests

json_min_file = Path(
    "/home/tuyen/Privat/Seafile/SSC/Opencalls/2025/ECPO/exchange-materials/100_jb_labelstudio.json"
)
output_dir = Path("/home/tuyen/SSC_repos/ecpo-eynollah/data/jb_org")
output_dir.mkdir(parents=True, exist_ok=True)

with open(json_min_file, "r") as f:
    data = json.load(f)

total = len(data)
print(f"Total images to download: {total}")

downloaded = 0
failed = 0
exist = 0
for item in data:
    img_link = item["image"]
    img_name = f"{item['name']}.png"

    # download image
    response = requests.get(img_link)
    if response.status_code == 200:
        if (output_dir / img_name).exists():
            exist += 1
            print(f"Image {img_name} already exists. Adding suffix '_dup'.")
            img_name = f"{item['name']}_dup.png"

        with open(output_dir / img_name, "wb") as img_file:
            img_file.write(response.content)
            downloaded += 1
        print(f"Downloaded {img_name}")
    else:
        failed += 1
        print(f"Failed to download {img_name} from {img_link}")

print(f"Download completed: {downloaded} succeeded, {failed} failed.")
print(f"{exist} images already existed and were added with suffix '_dup'.")
