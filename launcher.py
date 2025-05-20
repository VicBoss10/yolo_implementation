from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # <--- ¡Agrega esta línea!

@app.route('/start', methods=['POST'])
def start_stream():
    data = request.json
    nombre_camara = data.get("nombre_camara")
    if not nombre_camara:
        return jsonify({"error": "No se recibió el nombre de la cámara"}), 400

    # Guardar el nombre en el archivo
    with open("selected_camera.txt", "w") as f:
        f.write(nombre_camara.strip())

    # Lanzar app.py (en segundo plano)
    subprocess.Popen(["python3", "app.py"])
    return jsonify({"status": "stream iniciado"}), 200

if __name__ == "__main__":
    app.run(port=5001)