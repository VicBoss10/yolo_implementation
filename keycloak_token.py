import requests
import json

# Configuración de Keycloak
KEYCLOAK_URL = "http://localhost:8080/realms/detec-realm/protocol/openid-connect/token"
CLIENT_ID = "yolo-client"
CLIENT_SECRET = "TKFCY6RUMFGEsw0v4XRePtnAsPEMaWQP"  # Reemplaza con el client_secret real
BACKEND_URL = "http://localhost:8081/devices"  # Ajusta según la ruta real de tu endpoint

def get_keycloak_token():
    """Obtiene un token de Keycloak usando Client Credentials Grant."""
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    response = requests.post(KEYCLOAK_URL, data=payload)
    response.raise_for_status()  # Lanza una excepción si la petición falla
    return response.json()["access_token"]

def create_device():
    """Crea un dispositivo enviando los datos al backend."""
    # Crear el objeto Device (ajusta los valores según tus necesidades)
    device = {
        "name": "Camera_001",
        "type": "IP_CAMERA",
        "location": "Main Entrance",
        "status": "ACTIVE"  # Debe coincidir con DeviceStatus (ACTIVE o INACTIVE)
    }

    # Obtener el token
    token = get_keycloak_token()

    # Enviar la petición al backend
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(BACKEND_URL, json=device, headers=headers)

    # Manejar la respuesta
    if response.status_code == 200 or response.status_code == 201:
        return response.json()  # Devuelve el dispositivo creado
    else:
        raise Exception(f"Error al crear el dispositivo: {response.status_code} - {response.text}")

# Ejemplo de uso
try:
    result = create_device()
    print("Dispositivo creado:", result)
except Exception as e:
    print("Error:", e)