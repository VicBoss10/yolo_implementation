from ultralytics import YOLO
from services.keycloak_token import send_detection_to_backend  # Importa la función reutilizable

# Cargar el modelo de YOLO
model = YOLO("yolo11n.pt")

# Especificar el path de las imágenes
source = ["./inputs/4.jpg"]

def process_images():
    """Procesa las imágenes y envía las detecciones al backend."""
    results = model(source)

    for result, image_path in zip(results, source):
        print(f"\nProcesando imagen: {image_path}")
        print("------")

        # Extraer detecciones
        detected_objects = []
        for box in result.boxes:
            label = result.names[int(box.cls)]
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

        print(f"Detecciones encontradas: {len(detected_objects)}")
        for obj in detected_objects:
            print(f" - {obj['label']} (Confianza: {obj['confidence']:.2f}, Coordenadas: [{obj['x1']},{obj['y1']}]-[{obj['x2']},{obj['y2']}])")

        # Enviar detecciones al backend si hay objetos
        if detected_objects:
            send_detection_to_backend(detected_objects)
        else:
            print(f"No se encontraron objetos en {image_path}")

        # Mostrar la imagen con detecciones (opcional)
        result.show()

if __name__ == "__main__":
    process_images()