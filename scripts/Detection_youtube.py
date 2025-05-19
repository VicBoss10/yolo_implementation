from ultralytics import YOLO
import cv2
import imutils

# Cargar el modelo YOLO
model = YOLO("yolo11n.pt")

# Abrir el video (puedes cambiar la ruta o usar 0 para webcam)
source = ("https://www.youtube.com/watch?v=6wEnwHvYibc")

results = model(source, stream=True)

for result in results:
    annotated_frame = result.plot()
    annotated_frame = imutils.resize(annotated_frame, width=800)  # Redimensionar el frame para mostrarlo

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()