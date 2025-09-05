import os
from pathlib import Path

# === Settings ===
images_dir = Path("ds\img")   # folder with images
labels_dir = Path("ds\lbs")   # folder with labels .txt
prefix = "img_"               # prefix for new files names
exts = [".jpg"]               # format for files
start_idx = 0               # since this position/index
pad = 3                              

# === Script ===
images = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in exts]
images.sort()

for i, img_name in enumerate(images, start=start_idx):
    old_img_path = images_dir / img_name
    stem_new = f"{prefix}{str(i).zfill(pad)}"
    new_img_path = images_dir / f"{stem_new}{old_img_path.suffix.lower()}"
    
    # Changing name fore images
    os.rename(old_img_path, new_img_path)
    
    # Changing txt, if it exists
    old_txt_path = labels_dir / f"{Path(img_name).stem}.txt"
    new_txt_path = labels_dir / f"{stem_new}.txt"
    if old_txt_path.exists():
        os.rename(old_txt_path, new_txt_path)

print("Everything is ready! All images and txt are renamed.")
