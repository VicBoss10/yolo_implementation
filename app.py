from flask import Flask, Response
from yolo_service import generar_frames, liberar_recursos
import signal
import sys

app = Flask(__name__)

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

if __name__ == "__main__":
    print("Servidor Flask corriendo en http://localhost:5000/video")
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        liberar_recursos()