import time
import requests
import json

# Configuración de Keycloak y backend
KEYCLOAK_URL = "http://localhost:8080/realms/detec-realm/protocol/openid-connect/token"
CLIENT_ID = "yolo-client"
CLIENT_SECRET = "TKFCY6RUMFGEsw0v4XRePtnAsPEMaWQP"
BACKEND_URL = "http://localhost:8081/detections"
DEVICE_API_URL = "http://localhost:8081/devices/devicename/{}"

_token_cache = {"token": None, "expires_at": 0}
_device_id_cache = None

def get_keycloak_token():
    """Obtiene y cachea el token de Keycloak."""
    global _token_cache
    if _token_cache["token"] and _token_cache["expires_at"] > time.time():
        return _token_cache["token"]
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    try:
        response = requests.post(KEYCLOAK_URL, data=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 60)
        return _token_cache["token"]
    except Exception as e:
        print(f"Error al obtener el token: {e}")
        return None

def get_device_id_from_api():
    """Obtiene y cachea el device_id."""
    global _device_id_cache
    if _device_id_cache:
        return _device_id_cache
    try:
        with open("selected_camera.txt", "r") as f:
            nombre_camara = f.read().strip()
        url = DEVICE_API_URL.format(nombre_camara)
        token = get_keycloak_token()
        if not token:
            print("No se pudo obtener el token para consultar DEVICE_ID")
            return None
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        _device_id_cache = data.get("id")
        return _device_id_cache
    except Exception as e:
        print(f"Error al consultar DEVICE_ID en el API: {e}")
        return None

def send_detection_to_backend(detected_objects):
    """Envía una detección al backend."""
    device_id = get_device_id_from_api()
    print(f"DEVICE_ID obtenido: {device_id}")
    if not device_id:
        print("No se pudo obtener el DEVICE_ID")
        return False
    detection = {
        "device": {"id": device_id},
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