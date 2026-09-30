# Hass Avocado Ripeness Classification

This project classifies Hass avocado ripeness from camera images. It includes a dataset-labeling script, a trained YOLO classification model, and a live camera interface built with Flet.

## Workflow

### 1. Acquire the dataset

Download the **'Hass' Avocado Ripening Photographic Dataset** from Mendeley Data (citation below). The classification model used by this project was trained on this dataset.

`labeling.py` expects the spreadsheet `Avocado Ripening Dataset.xlsx` in the project root and the original images under:

```text
Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/
```

### 2. Label and split the images

The spreadsheet provides each image's `File Name` and `Ripening Index Classification`. Run the labeling script from the project root:

```bash
python labeling.py
```

The script moves the source images into `classified/` and creates reproducible training, validation, and test splits (80%, 10%, and 10%; random seed 42). Each split is organized into these five dataset classes:

- `Underripe`
- `Breaking`
- `Ripe_first_stage`
- `Ripe_second_stage`
- `Overripe`

**Keep a backup of the original dataset:** the script moves files out of the source directory. The generated layout is:

```text
classified/
├── train/
├── val/
└── test/
```

### 3. Train the classification model

The model classifies images into the five ripeness classes above. The trained model used for inference is provided at `model/best_hass.pt`. This repository does not include the model-training script or its training configuration. Use the prepared train and validation splits in your training workflow, and reserve the test split for final evaluation.

### 4. Install dependencies and run the interface

Install the dependencies and launch the Flet application from the project root:

```bash
python -m pip install -r requirements.txt
python interface.py
```

`interface.py` opens the camera, while `detection.py` loads `model/best_hass.pt` and classifies each frame. The interface displays the predicted ripeness class and confidence. The configured OpenCV camera index is `1` (`CAMERA_INDEX` in `interface.py`); change it if your camera uses another index. Run the application from the project root so the relative model path resolves.

### 5. Set camera exposure and illumination

The camera automatically adjusts exposure, making it difficult to maintain controlled illumination. On Linux, `v4l2-ctl` can be used to configure exposure manually. For example, these commands target `/dev/video2`:

```bash
v4l2-ctl -d /dev/video2 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_time_absolute=80
v4l2-ctl -d /dev/video2 --set-ctrl=exposure_dynamic_framerate=0
```

Control names and supported values vary by camera and driver. Inspect the controls supported by your device with:

```bash
v4l2-ctl -d /dev/video2 --list-ctrls
```

The application also sets exposure and focus through OpenCV in `interface.py`. The V4L2 device path and OpenCV camera index are different identifiers and may need adjustment for your setup. Keep camera position and illumination consistent when capturing images.

## Image region of interest

The dataset images are 800 × 800 pixels. For live inference, `interface.py` crops an 800 × 800-pixel region from each camera frame using `raw_frame[140:940, 560:1360]`, then sends that region to the classifier. Ensure the camera frame is large enough for the crop and that the model receives the same region of interest (ROI) used during training.

## Dataset citation

This project uses the following dataset to train its classification algorithm:

Xavier, Pedro; Rodrigues, Pedro; L. M. Silva, Cristina (2024). **“'Hass' Avocado Ripening Photographic Dataset.”** *Mendeley Data*, V1. [https://doi.org/10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1)
