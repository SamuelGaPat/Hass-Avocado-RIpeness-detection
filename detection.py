from ultralytics import YOLO
import cv2

class Avocado_detector:

    def __init__(self):

        self.model = YOLO("model/best_hass.pt")

        # Your desired labels
        self.class_names = {
            0: "Breaking",
            1: "Ripe_first_stage",
            2: "Ripe_second_stage",
            3: "Underripe",
            4: "Overripe"
        }

    def classify(self,frame):

        results = self.model(frame, verbose=False)

        result = results[0]

        # Get predicted class ID
        top1 = result.probs.top1

        # Get confidence
        top1_conf = result.probs.top1conf.item()

        # Use YOUR mapping instead of result.names

        top1_name = self.class_names[top1]

        # text = f"{top1_name}: {top1_conf:.2f}"

        # cv2.putText(
        #     frame,
        #     text,
        #     (20, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.2,
        #     (0, 255, 0),
        #     3
        # )

        return frame, top1_name, top1_conf

