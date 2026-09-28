import pandas as pd
import os
import shutil

original_folder = "Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset"

folders = {
    "1": "Underripe",
    "2": "Breaking",
    "3": "Ripe_first_stage",
    "4": "Ripe_second_stage",
    "5": "verripe"
}

df = pd.read_excel("Avocado Ripening Dataset.xlsx")

train = df.sample(frac=0.8, random_state=42)

remaining = df.drop(train.index)

test = remaining.sample(frac=0.5, random_state=42)

validate = remaining.drop(test.index)

splits = [train, test, validate ]
folder_splits = ['classified/train','classified/test','classified/val']

for dfr, folder in zip(splits,folder_splits):

    for index, row in dfr.iterrows():

        filename = str(row["File Name"])
        classification = str(row["Ripening Index Classification"])

        source = os.path.join(original_folder, filename + ".jpg")
        destination_folder = os.path.join(folder,folders[classification])
        destination = os.path.join(destination_folder, filename + ".jpg")

        os.makedirs(destination_folder, exist_ok=True)

        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"Moved: {filename}.jpg -> {destination_folder}")
        else:
            print(f"File not found: {source}")