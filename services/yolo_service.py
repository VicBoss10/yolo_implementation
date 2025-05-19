from ultralytics import YOLO
import cv2
import time
from services.keycloak_token import send_detection_to_backend
import subprocess
import re

def get_camera_index_by_base_name(camera_base_name):
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.splitlines()
    index = -1
    for i, line in enumerate(lines):
        # Extraer la parte base antes del primer paréntesis o dos puntos
        base = re.split(r'[:(]', line)[0].strip()
        if camera_base_name.strip() == base:
            # Buscar la línea siguiente que contiene /dev/videoX
            for j in range(i+1, len(lines)):
                match = re.search(r'/dev/video(\d+)', lines[j])
                if match:
                    index = int(match.group(1))
                    return index
    return None

# Leer el nombre de la cámara desde el archivo
with open("selected_camera.txt", "r") as f:
    nombre_base = f.read().strip()

indice = get_camera_index_by_base_name(nombre_base)
if indice is None:
    raise RuntimeError(f"No se encontró la cámara con nombre base: {nombre_base}")

# Configuración para comparación de detecciones
MIN_INTERVAL_SECONDS = 5  # Intervalo mínimo entre envíos
COORDINATE_THRESHOLD = 50  # Umbral para cambio en coordenadas (píxeles)
CONFIDENCE_THRESHOLD = 0.1  # Umbral para cambio en confianza

# Variables globales
last_sent_time = 0
previous_detections = []

# Cargar el modelo YOLO
model = YOLO("yolo11n.pt")

# Inicializar cámara
print(f"Abriendo cámara con índice: {indice}")
cap = cv2.VideoCapture(indice)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

# Establecer resolución 16:9 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def liberar_recursos():
    """Libera la cámara."""
    print("Liberando cámara...")
    cap.release()
    cv2.destroyAllWindows()

def are_detections_different(current_detections, previous_detections):
    """Compara detecciones para determinar si son significativamente diferentes."""
    global last_sent_time

    current_time = time.time()
    if current_time - last_sent_time < MIN_INTERVAL_SECONDS:
        return False

    if not previous_detections and current_detections:
        return True

    if len(current_detections) != len(previous_detections):
        
        return True

    for curr, prev in zip(current_detections, previous_detections):
        if curr["label"] != prev["label"]:
            return True
        if (abs(curr["x1"] - prev["x1"]) > COORDINATE_THRESHOLD or
            abs(curr["y1"] - prev["y1"]) > COORDINATE_THRESHOLD or
            abs(curr["x2"] - prev["x2"]) > COORDINATE_THRESHOLD or
            abs(curr["y2"] - prev["y2"]) > COORDINATE_THRESHOLD):
            return True
        if abs(curr["confidence"] - prev["confidence"]) > CONFIDENCE_THRESHOLD:
            return True

    return False

def generar_frames():
    """Genera frames con detecciones YOLO y envía al backend si es necesario."""
    global previous_detections, last_sent_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara")
            break

        # Realizar predicción con YOLO
        results = model.predict(source=frame, show=False, stream=False)
        annotated_frame = results[0].plot()

        # Extraer detecciones
        detected_objects = []
        for box in results[0].boxes:
            label = results[0].names[int(box.cls)]
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        # Comparar y enviar detecciones si son diferentes
        if detected_objects and are_detections_different(detected_objects, previous_detections):
            if send_detection_to_backend(detected_objects):
                last_sent_time = time.time()
                previous_detections = detected_objects.copy()

        # Generar frame para el stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    