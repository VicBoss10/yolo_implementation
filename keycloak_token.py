import requests
import json

# Configuración de Keycloak y backend
KEYCLOAK_URL = "http://localhost:8080/realms/detec-realm/protocol/openid-connect/token"
CLIENT_ID = "yolo-client"
CLIENT_SECRET = "TKFCY6RUMFGEsw0v4XRePtnAsPEMaWQP"
BACKEND_URL = "http://localhost:8081/detections"
DEVICE_ID = 1

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