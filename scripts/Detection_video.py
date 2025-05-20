from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO 
model = YOLO("yolo11n.pt")

video_path = "./inputs/1.mp4" 

cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
wait_time = int(1000 / fps) if fps > 0 else 30  # tiempo en ms

# Definir el codec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos la inferencia de YOLO
    results = model(frame)

    # Obtener el frame anotado
    annotated_frame = results[0].plot()

    # Escribir el frame anotado en el video de salida
    out.write(annotated_frame)

    # Mostrar resultados directamente usando Ultralytics
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()