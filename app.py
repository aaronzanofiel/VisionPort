import cv2
import numpy as np
import base64
from ultralytics import YOLO


class TrafficDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

        self.vehicle_classes = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    def detect_vehicles(self, image_bytes):

        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        results = self.model.predict(
            source=image,
            conf=0.35,
            verbose=False
        )

        annotated_image = image.copy()

        vehicle_counts = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0
        }

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in self.vehicle_classes:
                    continue

                label = self.vehicle_classes[class_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                vehicle_counts[label] += 1

                detections.append({
                    "vehicle": label,
                    "confidence": round(confidence * 100, 2),
                    "box": [x1, y1, x2, y2]
                })

                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 120),
                    2
                )

                cv2.putText(
                    annotated_image,
                    f"{label} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 120),
                    2
                )

        total_vehicles = sum(vehicle_counts.values())

        if total_vehicles < 10:
            density = "Low"
        elif total_vehicles < 25:
            density = "Medium"
        else:
            density = "High"

        success, encoded_image = cv2.imencode(
            ".jpg", annotated_image
        )

        if not success:
            raise RuntimeError("Image encoding failed")

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        return {
            "total_vehicles": total_vehicles,
            "vehicle_counts": vehicle_counts,
            "traffic_density": density,
            "detections": detections,
            "annotated_image": image_base64
        }


# Initialize detector once when needed
_detector = None


def get_detector():

    global _detector

    if _detector is None:
        _detector = TrafficDetector()

    return _detector


def process_traffic_image(image_bytes):

    detector = get_detector()

    return detector.detect_vehicles(image_bytes)