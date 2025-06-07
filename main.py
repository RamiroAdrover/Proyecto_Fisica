import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
=======
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go

# Importar threading para agregar cronometro al video
import threading

# Funciones de análisis
def smooth_data(data, window_length=7, polyorder=2):
    """Suaviza los datos usando filtro Savitzky-Golay"""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def calculate_derivatives_with_spacing(df, spacing=3):
    """Calcula velocidad y aceleración usando diferencias con espaciado"""
    # Crear copia del dataframe
    df_smooth = df.copy()
    
    # Suavizar posición
    df_smooth['x_m_smooth'] = smooth_data(df['x_m'].values)
    df_smooth['y_m_smooth'] = smooth_data(df['y_m'].values)
    
    # Calcular velocidad con espaciado
    df_smooth['vx_calculated'] = df_smooth['x_m_smooth'].diff(spacing) / (spacing / 60)  # Asumiendo 60 FPS
    df_smooth['vy_calculated'] = df_smooth['y_m_smooth'].diff(spacing) / (spacing / 60)
    
    # Suavizar velocidad
    df_smooth['vx_smooth'] = smooth_data(df_smooth['vx_calculated'].fillna(0).values)
    df_smooth['vy_smooth'] = smooth_data(df_smooth['vy_calculated'].fillna(0).values)
    
    # Calcular aceleración con espaciado
    df_smooth['ax_calculated'] = df_smooth['vx_smooth'].diff(spacing) / (spacing / 60)
    df_smooth['ay_calculated'] = df_smooth['vy_smooth'].diff(spacing) / (spacing / 60)
    
    return df_smooth

def find_critical_time(df):
    """Encuentra el tiempo crítico donde la velocidad en Y es máxima"""
    if 'vy_smooth' in df.columns:
        max_idx = df['vy_smooth'].idxmax()
        return df.loc[max_idx, 'nro_frame'], max_idx
    return None, None

def linear_function(x, a, b):
    """Función lineal para ajuste: y = ax + b"""
    return a * x + b

def analyze_free_fall(df, critical_frame):
    """Analiza la caída libre después del punto crítico"""
    if critical_frame is None:
        return None, None
    
    # Datos después del punto crítico
    free_fall_data = df[df['nro_frame'] >= critical_frame].copy()
    
    if len(free_fall_data) < 5:
        return None, None
    
    # Ajustar línea recta a la velocidad en Y durante caída libre
    x_data = free_fall_data['nro_frame'].values
    y_data = free_fall_data['vy_smooth'].values
    
    try:
        popt, pcov = curve_fit(linear_function, x_data, y_data)
        gravity_estimate = abs(popt[0]) * 60  # Convertir a m/s² considerando FPS
        return popt, gravity_estimate
    except:
        return None, None

def predict_trajectory(x0, y0, vx0, vy0, t_max=2.0, dt=0.01, g=9.81):
    """Predice la trayectoria usando ecuaciones de movimiento projectil"""
    t = np.arange(0, t_max, dt)
    x_pred = x0 + vx0 * t
    y_pred = y0 + vy0 * t - 0.5 * g * t**2
    
    # Solo valores positivos de y
    valid_idx = y_pred >= 0
    return x_pred[valid_idx], y_pred[valid_idx], t[valid_idx]
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    "velocity": {"pos": (10, 30), "size": (20, 20), "label": "Velocidad", "state": True},
    "acceleration": {"pos": (10, 60), "size": (20, 20), "label": "Aceleración", "state": True},
    "trajectory": {"pos": (10, 90), "size": (20, 20), "label": "Trayectoria", "state": True},
    "magnitudes": {"pos": (10, 120), "size": (20, 20), "label": "Magnitudes", "state": True},
=======
    "velocity": {"pos": (10, 60), "size": (20, 20), "label": "Velocidad Basica", "state": False},
    "acceleration": {"pos": (10, 90), "size": (20, 20), "label": "Aceleracion Basica", "state": False},
    "magnitudes": {"pos": (10, 120), "size": (20, 20), "label": "Magnitudes", "state": True},
    "x_components": {"pos": (10, 150), "size": (20, 20), "label": "Componentes X", "state": False},
    "prediction": {"pos": (10, 180), "size": (20, 20), "label": "Predicción", "state": False},
    "smooth_vectors": {"pos": (10, 210), "size": (20, 20), "label": "Vectores Suavizados", "state": True},
    "y_components": {"pos": (10, 240), "size": (20, 20), "label": "Solo Componentes Y", "state": True},
>>>>>>> Stashed changes
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
prev_y_m = None
prev_time = None
velocity_y_m = 0
acceleration_y_m = 0

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
            adjusted_y_m = adjusted_y_px * pixels_to_meters
            adjusted_x_m = adjusted_x_px * pixels_to_meters

            # Calcula el tiempo actual
            current_time = time.time()

            if prev_y_m is not None and prev_time is not None:
                # Calcula la velocidad en el eje Y (m/s)
                velocity_y_m = (adjusted_y_m - prev_y_m) / (current_time - prev_time)

                # Calcula la aceleración en el eje Y (m/s²)
                acceleration_y_m = (velocity_y_m - prev_velocity_y_m) / (current_time - prev_time)

                # Predice la posición futura
                delta_t = current_time - prev_time
                predicted_y_m = adjusted_y_m + velocity_y_m * delta_t + 0.5 * acceleration_y_m * (delta_t ** 2)
                predicted_x_m = adjusted_x_m  # Suponemos que no hay aceleración en X

                # Convierte la posición predicha a píxeles
                predicted_y_px = int(frame_height - (predicted_y_m / pixels_to_meters))
                predicted_x_px = int(predicted_x_m / pixels_to_meters)

                # Dibuja la posición predicha en el video
                cv2.circle(frame, (predicted_x_px, predicted_y_px), 5, (0, 0, 255), -1)  # Punto rojo para la predicción

            # Dibuja el rectángulo del objeto rastreado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dibuja los vectores si están habilitados
            if checkboxes["velocity"]["state"]:
                # Dibuja el vector de velocidad (flecha azul)
                velocity_arrow_end = (adjusted_x_px, int(frame_height - ((adjusted_y_m + velocity_y_m * 0.5) / pixels_to_meters)))
                cv2.arrowedLine(frame, (adjusted_x_px, frame_height - adjusted_y_px), velocity_arrow_end, (255, 0, 0), 2)

            if checkboxes["acceleration"]["state"]:
                # Dibuja el vector de aceleración (flecha roja)
                acceleration_arrow_end = (adjusted_x_px, int(frame_height - ((adjusted_y_m + acceleration_y_m * 0.05) / pixels_to_meters)))
                cv2.arrowedLine(frame, (adjusted_x_px, frame_height - adjusted_y_px), acceleration_arrow_end, (0, 0, 255), 2)

            # Muestra las magnitudes si están habilitadas
            if checkboxes["magnitudes"]["state"]:
                cv2.putText(frame, f"Y: {adjusted_y_m:.1f} m", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Vel: {velocity_y_m:.1f} m/s", (frame_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Acel: {acceleration_y_m:.1f} m/s^2", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Agrega la posición actual a la trayectoria
            trajectory.append((adjusted_x_px, frame_height - adjusted_y_px))

            # Dibuja la trayectoria si está habilitada
            if checkboxes["trajectory"]["state"]:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 255), 2)

            # Actualiza las variables previas
            prev_y_m = adjusted_y_m
            prev_time = current_time
            prev_velocity_y_m = velocity_y_m

            # Guarda los datos en la lista para exportar luego
            data.append({
                "nro_frame": frame_count,
                "x_m": round(adjusted_x_m, 2),
                "y_m": round(adjusted_y_m, 2),
                "vy_m/s": round(velocity_y_m, 2),
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

    # Cronómetro en segundos
    if not paused:
        cv2.putText(frame, f"Tiempo: {current_time:.2f} s", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


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
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Posición Y vs frame
axs[0].plot(df["nro_frame"], df["y_m"], label="Posición Y (m)", color="blue")
axs[0].set_ylabel("Posición Y (m)")
axs[0].grid(True)
axs[0].legend()

# Velocidad Y vs frame
axs[1].plot(df["nro_frame"], df["vy_m/s"], label="Velocidad Y (m/s)", color="green")
axs[1].set_ylabel("Velocidad Y (m/s)")
axs[1].grid(True)
axs[1].legend()

# Aceleración Y vs frame
axs[2].plot(df["nro_frame"], df["ay_m/s^2"], label="Aceleración Y (m/s²)", color="red")
axs[2].set_xlabel("Número de Frame")
axs[2].set_ylabel("Aceleración Y (m/s²)")
axs[2].grid(True)
axs[2].legend()

# Guardar gráfico y mostrar
plt.tight_layout()
plt.savefig("graficos_dinamica.png")
plt.show()
