from flask import Flask, Response
from ultralytics import YOLO
import cv2
import signal
import sys
import requests
import json
from datetime import datetime
import time

app = Flask(__name__)

# Configuración de Keycloak
KEYCLOAK_URL = "http://localhost:8080/realms/detec-realm/protocol/openid-connect/token"
CLIENT_ID = "yolo-client"
CLIENT_SECRET = "TKFCY6RUMFGEsw0v4XRePtnAsPEMaWQP"  # Reemplaza con el client_secret
BACKEND_URL = "http://localhost:8081/detections"
DEVICE_ID = 1  # ID del dispositivo existente (ajusta según tu base de datos)


# Configuración para comparación de detecciones
MIN_INTERVAL_SECONDS = 5  # Intervalo mínimo entre envíos
COORDINATE_THRESHOLD = 50  # Umbral para cambio en coordenadas (píxeles)
CONFIDENCE_THRESHOLD = 0.1  # Umbral para cambio en confianza
last_sent_time = 0
previous_detections = []  # Almacena las detecciones anteriores

# Cargar el modelo YOLO
model = YOLO("yolo11n.pt")

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

# Establecer resolución 16:9 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def liberar_recursos(signal_received=None, frame=None):
    print("\nLiberando cámara y cerrando servidor...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Manejar interrupciones (Ctrl+C o cierre)
signal.signal(signal.SIGINT, liberar_recursos)
signal.signal(signal.SIGTERM, liberar_recursos)

def get_keycloak_token():
    """Obtiene un token de Keycloak usando Client Credentials Grant."""
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(KEYCLOAK_URL, data=payload, timeout=5)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener el token: {e}")
        return None

def send_detection_to_backend(detected_objects):
    """Envía una detección al backend."""
    detection = {
        "device": {"id": DEVICE_ID},
        "detectedObjects": detected_objects
    }
    try:
        token = get_keycloak_token()
        if not token:
            print("No se pudo obtener el token")
            return False
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(BACKEND_URL, json=detection, headers=headers, timeout=5)
        response.raise_for_status()
        print("Detección enviada exitosamente:", response.json())
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP al enviar detección: {response.status_code} - {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar detección: {e}")
        return False

def are_detections_different(current_detections, previous_detections):
    """Compara detecciones para determinar si son significativamente diferentes."""
    global last_sent_time

    # Verificar si han pasado al menos MIN_INTERVAL_SECONDS
    current_time = time.time()
    if current_time - last_sent_time < MIN_INTERVAL_SECONDS:
        return False

    # Si no hay detecciones previas y hay nuevas, son diferentes
    if not previous_detections and current_detections:
        return True

    # Si el número de objetos cambió, son diferentes
    if len(current_detections) != len(previous_detections):
        return True

    # Comparar cada detección
    for curr, prev in zip(current_detections, previous_detections):
        # Diferencia en etiqueta
        if curr["label"] != prev["label"]:
            return True

        # Diferencia en coordenadas
        if (abs(curr["x1"] - prev["x1"]) > COORDINATE_THRESHOLD or
            abs(curr["y1"] - prev["y1"]) > COORDINATE_THRESHOLD or
            abs(curr["x2"] - prev["x2"]) > COORDINATE_THRESHOLD or
            abs(curr["y2"] - prev["y2"]) > COORDINATE_THRESHOLD):
            return True

        # Diferencia en confianza
        if abs(curr["confidence"] - prev["confidence"]) > CONFIDENCE_THRESHOLD:
            return True

    return False

def generar_frames():
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
            label = results[0].names[int(box.cls)]  # Etiqueta (ej. "person")
            confidence = float(box.conf)  # Confianza
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas
            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        # Comparar con detecciones anteriores
        if detected_objects and are_detections_different(detected_objects, previous_detections):
            if send_detection_to_backend(detected_objects):
                last_sent_time = time.time()
                previous_detections = detected_objects.copy()  # Actualizar detecciones anteriores

        # Generar frame para el stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Servidor Flask corriendo en http://localhost:5000/video")
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        liberar_recursos()