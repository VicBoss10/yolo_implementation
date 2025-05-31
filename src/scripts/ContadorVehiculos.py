from ultralytics import YOLO
import cv2
import time
import os

# Cargar el modelo YOLO 
model = YOLO("yolo11n.pt")

video_path = "./inputs/Vehiculos.MP4" 

# Define las clases de vehículos según el modelo (pueden variar)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 0 else 30

# Definir la posición de la línea de conteo (por ejemplo, a la mitad de la imagen)
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    exit()
height, width, _ = frame.shape
line_y = height // 2
line_x = (width // 2)-7

vehicle_count = 0
recent_centers = []  # Lista de (x, y, timestamp)
DIST_THRESHOLD = 50  # píxeles de tolerancia en X para considerar que es el mismo vehículo
TIME_THRESHOLD = 1.0  # segundos para considerar un centro como "reciente"

# --- NUEVO: Contadores sube/baja ---
if 'count_up' not in locals():
    count_up = 0
    count_down = 0
    recent_crosses = []

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "vehiculos_procesado.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' para .avi
out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25, (width, height))

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    # Elimina centros viejos
    recent_centers = [(x, y, t) for (x, y, t) in recent_centers if now - t < TIME_THRESHOLD]

    results = model(frame)
    boxes = results[0].boxes
    names = results[0].names

    vehicle_frame = frame.copy()
    # Dibuja la línea horizontal de conteo
    cv2.line(vehicle_frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    # Dibuja la línea vertical para dividir los carriles
    #line_x = (width // 2)-10
    #cv2.line(vehicle_frame, (line_x, 0), (line_x, height), (255, 0, 0), 2)

    for i, box in enumerate(boxes):
        cls_id = int(box.cls)
        label = names[cls_id]
        if label in VEHICLE_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)
            cv2.rectangle(vehicle_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(vehicle_frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(vehicle_frame, (center_x, center_y), 4, (255, 0, 0), -1)
            # Contar si cruza la línea y no ha sido contado antes
            if abs(center_y - line_y) < 5:
                if not any(abs(center_x - x) < DIST_THRESHOLD for (x, y, t) in recent_centers):
                    vehicle_count += 1
                    recent_centers.append((center_x, center_y, now))

    # --- NUEVO: Contadores sube/baja ---
    for i, box in enumerate(boxes):
        cls_id = int(box.cls)
        label = names[cls_id]
        if label in VEHICLE_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)
            # Solo cuando cruza la línea horizontal y no ha sido contado recientemente
            if abs(center_y - line_y) < 5:
                if not any(abs(center_x - x) < DIST_THRESHOLD for (x, y, t) in recent_crosses):
                    if center_x < line_x:
                        count_up += 1
                    else:
                        count_down += 1
                    recent_crosses.append((center_x, center_y, now))
    # Limpia cruces viejos
    recent_crosses = [(x, y, t) for (x, y, t) in recent_crosses if now - t < TIME_THRESHOLD]

    # Función helper para dibujar texto con fondo
    def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale=0.8, text_color=(255, 255, 255), 
                                bg_color=(0, 0, 0), thickness=2):
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calcular coordenadas del rectángulo de fondo
        padding = 5
        bg_rect = (
            position[0] - padding,
            position[1] - text_height - padding,
            text_width + (padding * 2),
            text_height + (padding * 2)
        )
        
        # Dibujar rectángulo semi-transparente
        overlay = img.copy()
        cv2.rectangle(overlay, (bg_rect[0], bg_rect[1]), 
                     (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
                     bg_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Dibujar texto
        cv2.putText(img, text, (position[0], position[1]), 
                    font, font_scale, text_color, thickness)

    # Configuración de la información
    margin_left = 20
    y_offset = 40
    spacing = 45

    # Panel de información
    panel_start = (10, 30)
    panel_width = 300
    panel_height = 180
    
    # Dibujar panel principal semi-transparente
    overlay = vehicle_frame.copy()
    cv2.rectangle(overlay, panel_start, 
                 (panel_start[0] + panel_width, panel_start[1] + panel_height), 
                 (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, vehicle_frame, 0.3, 0, vehicle_frame)

    # Dibujar textos con sus fondos
    draw_text_with_background(vehicle_frame, 
                            f"Total Vehiculos: {vehicle_count}", 
                            (margin_left, y_offset), 
                            bg_color=(50, 50, 50),
                            text_color=(255, 255, 255))
    
    draw_text_with_background(vehicle_frame, 
                            f"Carril Izquierdo: {count_up}", 
                            (margin_left, y_offset + spacing), 
                            bg_color=(50, 0, 0),
                            text_color=(255, 255, 255))
    
    draw_text_with_background(vehicle_frame, 
                            f"Carril Derecho: {count_down}", 
                            (margin_left, y_offset + spacing * 2), 
                            bg_color=(0, 50, 50),
                            text_color=(255, 255, 255))

    # Asegúrate de que el frame tenga el tamaño correcto
    vehicle_frame = cv2.resize(vehicle_frame, (width, height))
    out.write(vehicle_frame)  # Guarda el frame procesado

    cv2.imshow("YOLO Vehicle Detection & Counting", vehicle_frame)

    elapsed = (time.time() - start_time) * 1000  # en ms
    delay = max(1, int(wait_time - elapsed))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Libera el archivo de salida
cv2.destroyAllWindows()