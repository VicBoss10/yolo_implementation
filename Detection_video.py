from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO 
model = YOLO("yolo11n.pt")

video_path = "./inputs/bjhvkhj.MP4" 

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos la inferencia de YOLO
    results = model(frame)

    # Mostrar resultados directamente usando Ultralytics
    cv2.imshow("YOLO Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Liberar recursos
cap.release()
cv2.destroyAllWindows()