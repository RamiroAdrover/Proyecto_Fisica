import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# Diámetro de la pelota en metros
ball_diameter_m = 0.24

# Variables globales para el rectángulo
drawing = False
ix, iy = -1, -1
bbox = None
pixels_to_meters = None  # Relación píxeles a metros

# Lista para almacenar las posiciones de la trayectoria
trajectory = []

# Flags para mostrar/ocultar elementos
show_velocity = True
show_acceleration = True
show_trajectory = True
show_magnitudes = True

# Dimensiones de los checkboxes
checkboxes = {
    "velocity": {"pos": (10, 30), "size": (20, 20), "label": "Velocidad", "state": True},
    "acceleration": {"pos": (10, 60), "size": (20, 20), "label": "Aceleración", "state": True},
    "trajectory": {"pos": (10, 90), "size": (20, 20), "label": "Trayectoria", "state": True},
    "magnitudes": {"pos": (10, 120), "size": (20, 20), "label": "Magnitudes", "state": True},
    "x_components": {"pos": (10, 150), "size": (20, 20), "label": "Componentes X", "state": True},
}

# Función de callback para manejar clics del mouse
def mouse_callback(event, x, y, flags, param):
    global checkboxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for key, checkbox in checkboxes.items():
            cx, cy = checkbox["pos"]
            cw, ch = checkbox["size"]
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                checkbox["state"] = not checkbox["state"]

# Función de callback para dibujar el rectángulo
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Selecciona el objeto a rastrear", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x - ix, y - iy)
        cv2.destroyWindow("Selecciona el objeto a rastrear")

# Inicializa la captura de video
cap = cv2.VideoCapture("video/video.mp4")

# Lee el primer frame
ret, frame = cap.read()

# Ventana para seleccionar el objeto
cv2.namedWindow("Selecciona el objeto a rastrear")
cv2.setMouseCallback("Selecciona el objeto a rastrear", draw_rectangle)

while True:
    cv2.imshow("Selecciona el objeto a rastrear", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or bbox is not None:
        break

# Calcula la relación píxeles a metros usando el diámetro de la pelota
pixels_diameter = bbox[2]  # Ancho del rectángulo en píxeles
pixels_to_meters = ball_diameter_m / pixels_diameter

# Inicializa el tracker CSRT
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Dimensiones del video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Variables para calcular velocidad y aceleración
prev_x_m = None
prev_y_m = None
prev_time = None
velocity_x_m = 0
velocity_y_m = 0
prev_velocity_x_m = 0
prev_velocity_y_m = 0
acceleration_x_m = 0
acceleration_y_m = 0
offset_y_m = 1

# Variable para controlar la pausa
paused = False

# Lista para almacenar los datos del DataFrame
data = []

# Contador de frames
frame_count = 0

# Configurar el callback del mouse para la ventana principal
cv2.namedWindow("Rastreo CSRT")
cv2.setMouseCallback("Rastreo CSRT", mouse_callback)

while True:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("Fin del video o error al leer el frame.")
            break

        # Actualiza el tracker
        success, bbox = tracker.update(frame)

        if success:
            # Coordenadas del objeto rastreado
            x, y, w, h = [int(v) for v in bbox]

            # Ajusta el sistema de coordenadas: origen en la esquina inferior izquierda
            adjusted_y_px = frame_height - (y + h // 2)
            adjusted_x_px = x + w // 2

            # Convierte las coordenadas a metros
            adjusted_y_m = (adjusted_y_px * pixels_to_meters) - offset_y_m 
            adjusted_x_m = adjusted_x_px * pixels_to_meters            # Calcula el tiempo actual
            current_time = time.time()

            if prev_x_m is not None and prev_y_m is not None and prev_time is not None:
                # Calcula la velocidad en ambos ejes (m/s)
                delta_time = current_time - prev_time
                velocity_x_m = (adjusted_x_m - prev_x_m) / delta_time
                velocity_y_m = (adjusted_y_m - prev_y_m) / delta_time

                # Calcula la aceleración en ambos ejes (m/s²)
                acceleration_x_m = (velocity_x_m - prev_velocity_x_m) / delta_time
                acceleration_y_m = (velocity_y_m - prev_velocity_y_m) / delta_time

                # Predice la posición futura
                predicted_y_m = adjusted_y_m + velocity_y_m * delta_time + 0.5 * acceleration_y_m * (delta_time ** 2)
                predicted_x_m = adjusted_x_m + velocity_x_m * delta_time + 0.5 * acceleration_x_m * (delta_time ** 2)

                # Convierte la posición predicha a píxeles
                predicted_y_px = int(frame_height - (predicted_y_m / pixels_to_meters))
                predicted_x_px = int(predicted_x_m / pixels_to_meters)

                # Dibuja la posición predicha en el video
                cv2.circle(frame, (predicted_x_px, predicted_y_px), 5, (0, 0, 255), -1)  # Punto rojo para la predicción            # Dibuja el rectángulo del objeto rastreado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Dibuja los vectores si están habilitados
            center_x = adjusted_x_px
            center_y = frame_height - adjusted_y_px
            
            if checkboxes["velocity"]["state"] and prev_x_m is not None:
                # Escala para visualización de vectores (ajustable)
                velocity_scale = 50  # píxeles por m/s
                
                # Solo componente Y - flecha vertical
                velocity_end_x = center_x  # Sin componente X
                velocity_end_y = int(center_y - velocity_y_m * velocity_scale)  # Negativo porque Y crece hacia abajo en OpenCV
                
                # Dibuja el vector de velocidad (flecha azul) solo en Y
                cv2.arrowedLine(
                    frame,
                    (center_x, center_y),
                    (velocity_end_x, velocity_end_y),
                    (255, 0, 0),  # Azul
                    2,
                    tipLength=0.3
                )

            if checkboxes["acceleration"]["state"] and prev_x_m is not None:
                # Escala para visualización de aceleración (ajustable)
                acceleration_scale = 10  # píxeles por m/s²
                
                # Solo componente Y - flecha vertical hacia abajo (gravedad)
                accel_end_x = center_x  # Sin componente X
                accel_end_y = int(center_y + acceleration_y_m * acceleration_scale)  # Siempre hacia abajo
                
                # Dibuja el vector de aceleración (flecha roja) siempre hacia abajo
                cv2.arrowedLine(
                    frame,
                    (center_x, center_y),
                    (accel_end_x, accel_end_y),
                    (0, 0, 255),  # Rojo
                    2,
                    tipLength=0.3
                )

            # Dibuja vectores de flechas para las componentes en el eje X
            if checkboxes["x_components"]["state"] and prev_x_m is not None:
                # Componente X de velocidad (flecha verde claro)
                vel_x_end = (int(center_x + velocity_x_m * 50), center_y)
                cv2.arrowedLine(frame, (center_x, center_y), vel_x_end, (0, 255, 255), 2, tipLength=0.2)

                # Componente Y de velocidad (flecha amarilla)
                vel_y_end = (center_x, int(center_y - velocity_y_m * 50))
                cv2.arrowedLine(frame, (center_x, center_y), vel_y_end, (0, 255, 0), 2, tipLength=0.2)

            # Cambiar el color de la trayectoria en X
            if checkboxes["trajectory"]["state"]:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 165, 0), 2)  # Naranja para la trayectoria            # Muestra las magnitudes si están habilitadas
            if checkboxes["magnitudes"]["state"]:
                cv2.putText(frame, f"X: {adjusted_x_m:.1f} m", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Y: {adjusted_y_m:.1f} m", (frame_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if prev_x_m is not None:
                    cv2.putText(frame, f"Vx: {velocity_x_m:.1f} m/s", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Vy: {velocity_y_m:.1f} m/s", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Ax: {acceleration_x_m:.1f} m/s^2", (frame_width - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Ay: {acceleration_y_m:.1f} m/s^2", (frame_width - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Agrega la posición actual a la trayectoria
            trajectory.append((adjusted_x_px, frame_height - adjusted_y_px))

            # Dibuja la trayectoria si está habilitada
            if checkboxes["trajectory"]["state"]:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 165, 0), 2)  # Naranja para la trayectoria            # Actualiza las variables previas
            prev_x_m = adjusted_x_m
            prev_y_m = adjusted_y_m
            prev_time = current_time
            if prev_x_m is not None:
                prev_velocity_x_m = velocity_x_m
                prev_velocity_y_m = velocity_y_m

            # Guarda los datos en la lista para exportar luego
            data.append({
                "nro_frame": frame_count,
                "x_m": round(adjusted_x_m, 2),
                "y_m": round(adjusted_y_m, 2),
                "vx_m/s": round(velocity_x_m, 2),
                "vy_m/s": round(velocity_y_m, 2),
                "ax_m/s^2": round(acceleration_x_m, 2),
                "ay_m/s^2": round(acceleration_y_m, 2)
            })


        frame_count += 1

    # Dibujar los checkboxes
    for key, checkbox in checkboxes.items():
        cx, cy = checkbox["pos"]
        cw, ch = checkbox["size"]
        color = (0, 255, 0) if checkbox["state"] else (0, 0, 255)
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), color, -1)
        cv2.putText(frame, checkbox["label"], (cx + cw + 5, cy + ch - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Rastreo CSRT", frame)

    # Manejo de teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir
        break
    elif key == ord('p'):  # Pausar/Reanudar
        paused = not paused

cap.release()
cv2.destroyAllWindows()

# Crear un DataFrame con los datos recolectados
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv("resultados_rastreo.csv", index=False)
print("Datos guardados en 'resultados_rastreo.csv'")

# Generar gráficos de posición, velocidad y aceleración
fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

# Posición X e Y vs frame
axs[0].plot(df["nro_frame"], df["x_m"], label="Posición X (m)", color="orange")
axs[0].plot(df["nro_frame"], df["y_m"], label="Posición Y (m)", color="blue")
axs[0].set_ylabel("Posición (m)")
axs[0].grid(True)
axs[0].legend()

# Velocidad X e Y vs frame
axs[1].plot(df["nro_frame"], df["vx_m/s"], label="Velocidad X (m/s)", color="green")
axs[1].plot(df["nro_frame"], df["vy_m/s"], label="Velocidad Y (m/s)", color="red")
axs[1].set_ylabel("Velocidad (m/s)")
axs[1].grid(True)
axs[1].legend()

# Aceleración X e Y vs frame
axs[2].plot(df["nro_frame"], df["ax_m/s^2"], label="Aceleracion X (m/s²)", color="purple")
axs[2].plot(df["nro_frame"], df["ay_m/s^2"], label="Aceleracion Y (m/s²)", color="brown")
axs[2].set_ylabel("Aceleracion (m/s²)")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig("graficos_dinamica.png")
plt.show()
