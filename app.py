from flask import Flask, Response, request, jsonify
from flask_cors import CORS  
from services.yolo_service import generar_frames, liberar_recursos
import signal
import sys

app = Flask(__name__)
CORS(app)  

def signal_handler(signal_received, frame):
    """Maneja la interrupción para liberar recursos."""
    liberar_recursos()
    sys.exit(0)

# Manejar interrupciones (Ctrl+C o cierre)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/video')
def video():
    """Endpoint para streaming de video con detecciones."""
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    data = request.get_json()
    nombre_camara = data.get('nombre_camara')
    if not nombre_camara:
        return jsonify({"error": "No se proporcionó nombre_camara"}), 400
    with open("selected_camera.txt", "w") as f:
        f.write(nombre_camara.strip())
    return jsonify({"message": f"Cámara seleccionada: {nombre_camara}"}), 200

if __name__ == "__main__":
    print("Servidor Flask corriendo en http://localhost:500/video")
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        liberar_recursos()