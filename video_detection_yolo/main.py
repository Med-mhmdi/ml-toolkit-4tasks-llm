import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolov8n.pt")
    video_path = Path(__file__).parent / "samples" / "street.mp4"
    if not video_path.exists():
        print("⚠️ Please add a short video (street.mp4) to samples/")
        return
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
