from ultralytics import YOLO
import requests
import json

# Configuración de Keycloak
KEYCLOAK_URL = "http://localhost:8080/realms/detec-realm/protocol/openid-connect/token"
CLIENT_ID = "yolo-client"
CLIENT_SECRET = "TKFCY6RUMFGEsw0v4XRePtnAsPEMaWQP"  # Reemplaza con el client_secret
BACKEND_URL = "http://localhost:8081/detections"
DEVICE_ID = 1  # ID del dispositivo existente (ajusta según tu base de datos)

# Cargar el modelo de YOLO
model = YOLO("yolo11n.pt")

# Especificar el path de las imágenes
source = ["./inputs/CineBodega.jpg", "./inputs/IMG_7371.jpg", "./inputs/VN_9.jpg"]

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

def send_detection_to_backend(detected_objects, image_path):
    """Envía una detección al backend."""
    detection = {
        "device": {"id": DEVICE_ID},
        "detectedObjects": detected_objects
    }
    try:
        token = get_keycloak_token()
        if not token:
            print(f"No se pudo obtener el token para {image_path}")
            return False
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        print(f"Enviando detección para {image_path}:", json.dumps(detection, indent=2))
        response = requests.post(BACKEND_URL, json=detection, headers=headers, timeout=5)
        response.raise_for_status()
        print(f"Detección enviada exitosamente para {image_path}:", response.json())
        return True
    except requests.exceptions.HTTPError as e:
        print(f"Error HTTP al enviar detección para {image_path}: {response.status_code} - {response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar detección para {image_path}: {e}")
        return False

def process_images():
    """Procesa las imágenes y envía las detecciones al backend."""
    # Realizamos la inferencia de YOLO
    results = model(source)

    for result, image_path in zip(results, source):
        print(f"\nProcesando imagen: {image_path}")
        print("------")

        # Extraer detecciones
        detected_objects = []
        for box in result.boxes:
            label = result.names[int(box.cls)]  # Etiqueta (ej. "person")
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

        # Mostrar información de la detección
        print(f"Detecciones encontradas: {len(detected_objects)}")
        for obj in detected_objects:
            print(f" - {obj['label']} (Confianza: {obj['confidence']:.2f}, Coordenadas: [{obj['x1']},{obj['y1']}]-[{obj['x2']},{obj['y2']}])")

        # Enviar detecciones al backend si hay objetos
        if detected_objects:
            send_detection_to_backend(detected_objects, image_path)
        else:
            print(f"No se encontraron objetos en {image_path}")

        # Mostrar la imagen con detecciones (opcional)
        result.show()

if __name__ == "__main__":
    process_images()