from ultralytics import YOLO
import cv2

image = cv2.imread("./inputs/CineBodega.jpg")
image = cv2.resize(image, (1920, 1279))  # Redimensionar manteniendo relación 3:2

model = YOLO("yolo11n.pt")

# Realizamos la inferencia de YOLO
results = model(image)

# Mostrar resultados directamente usando Ultralytics
cv2.imshow("YOLO Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
