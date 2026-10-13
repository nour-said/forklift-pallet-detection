import sys
import time

import cv2
from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else 0   
model = YOLO("weights/best.pt")
cap = cv2.VideoCapture(source)

while True:
    start = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, conf=0.25, device="mps", verbose=False)[0]
    annotated = result.plot()                      

    fps = 1 / (time.time() - start)
    cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {len(result.boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Forklift & Pallet Detection (q to quit)", annotated)
    #time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()