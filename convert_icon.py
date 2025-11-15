from PIL import Image
import os

icon_path = "icon.png"
ico_path = "icon.ico"

if os.path.exists(icon_path):
    img = Image.open(icon_path)
    img.save(ico_path, 'ICO')
    print(f"Converted {icon_path} to {ico_path}")
else:
    print(f"{icon_path} not found")
