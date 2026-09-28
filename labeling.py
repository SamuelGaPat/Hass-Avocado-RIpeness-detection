import pandas as pd
import os
import shutil

original_folder = "Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset"

folders = {
    "1": "classified/Underripe",
    "2": "classified/Breaking",
    "3": "classified/Ripe_first_stage",
    "4": "classified/Ripe_second_stage",
    "5": "classified/Overripe"
}

df = pd.read_excel("Avocado Ripening Dataset.xlsx")

for index, row in df.iterrows():

    filename = str(row["File Name"])
    classification = str(row["Ripening Index Classification"])

    source = os.path.join(original_folder, filename + ".jpg")
    destination_folder = folders[classification]
    destination = os.path.join(destination_folder, filename + ".jpg")

    os.makedirs(destination_folder, exist_ok=True)

    if os.path.exists(source):
        shutil.move(source, destination)
        print(f"Moved: {filename}.jpg -> {destination_folder}")
    else:
        print(f"File not found: {source}")