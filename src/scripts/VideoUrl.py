import sys
from ultralytics import YOLO
import cv2
import time
import os

# === CONFIGURA TU VIDEO AQUÍ ===
source = "./inputs/v3.mp4"  # Ruta al video local o URL de YouTube
# source = "/home/victor/Documentos/Modelos/task/inputs/V2.mp4"

def get_youtube_stream(url):
    import yt_dlp
    ydl_opts = {
        'format': 'best[ext=mp4][protocol^=http]',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

if "youtube.com" in source or "youtu.be" in source:
    source = get_youtube_stream(source)

cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 0 else 30

ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    sys.exit()
height, width, _ = frame.shape

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "video_procesado.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25, (width, height))

model = YOLO("yolo11n.pt")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Asegura que el frame tenga el tamaño original antes de guardar
    if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
        annotated_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_LINEAR)

    out.write(annotated_frame)

    # Mostrar el frame (opcional, aquí sí puedes hacer resize si quieres)
    show_frame = cv2.resize(annotated_frame, (min(1280, width), min(720, height)))
    cv2.imshow("YOLO Detection", show_frame)

    elapsed = (time.time() - start_time) * 1000  # en ms
    delay = max(1, int(wait_time - elapsed))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video guardado en: {output_path}")