import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go

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
    "velocity": {"pos": (10, 30), "size": (20, 20), "label": "Velocidad Basica", "state": False},
    "acceleration": {"pos": (10, 60), "size": (20, 20), "label": "Aceleracion Basica", "state": False},
    "magnitudes": {"pos": (10, 90), "size": (20, 20), "label": "Magnitudes", "state": True},
    "x_components": {"pos": (10, 120), "size": (20, 20), "label": "Componentes X", "state": False},
    "prediction": {"pos": (10, 150), "size": (20, 20), "label": "Predicción", "state": False},
    "smooth_vectors": {"pos": (10, 180), "size": (20, 20), "label": "Vectores Suavizados", "state": True},
    "y_components": {"pos": (10, 210), "size": (20, 20), "label": "Solo Componentes Y", "state": True},
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

# Variables para análisis mejorado
smooth_velocity_x = 0
smooth_velocity_y = 0
smooth_acceleration_x = 0
smooth_acceleration_y = 0
predicted_trajectory_x = []
predicted_trajectory_y = []
initial_velocity_detected = False
initial_vx = 0
initial_vy = 0
initial_pos_x = 0
initial_pos_y = 0

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

        if success:            # Coordenadas del objeto rastreado
            x, y, w, h = [int(v) for v in bbox]
            
            # Ajusta el sistema de coordenadas: origen en la esquina inferior izquierda
            adjusted_y_px = frame_height - (y + h // 2)
            adjusted_x_px = x + w // 2

            # Convierte las coordenadas a metros
            adjusted_y_m = (adjusted_y_px * pixels_to_meters) - offset_y_m 
            adjusted_x_m = adjusted_x_px * pixels_to_meters
            
            # Calcula el tiempo actual
            current_time = 1 / cap.get(cv2.CAP_PROP_FPS) * frame_count

            if prev_x_m is not None and prev_y_m is not None and prev_time is not None:
                # Calcula la velocidad en ambos ejes (m/s) - método básico
                delta_time = current_time - prev_time
                velocity_x_m = (adjusted_x_m - prev_x_m) / delta_time
                velocity_y_m = (adjusted_y_m - prev_y_m) / delta_time

                # Calcula la aceleración en ambos ejes (m/s²) - método básico
                acceleration_x_m = (velocity_x_m - prev_velocity_x_m) / delta_time
                acceleration_y_m = (velocity_y_m - prev_velocity_y_m) / delta_time
                
                # MÉTODO MEJORADO: Calcular velocidades y aceleraciones usando espaciado (diff con -3)
                if len(data) >= 7:
                    # Crear DataFrame temporal para análisis con espaciado
                    df_temp = pd.DataFrame(data[-7:])  # Usar últimos 7 puntos
                    
                    # Aplicar suavizado Savitzky-Golay a las posiciones
                    if len(df_temp) >= 7:
                        x_smooth = smooth_data(df_temp['x_m'].values)
                        y_smooth = smooth_data(df_temp['y_m'].values)
                        
                        # Calcular velocidades usando diferencias con espaciado (como diff(-3))
                        spacing = 3  # Espaciado según sugerencia del texto
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        if len(x_smooth) > spacing:
                            # Velocidades mejoradas usando espaciado
                            vx_diff = x_smooth[-1] - x_smooth[-1-spacing]
                            vy_diff = y_smooth[-1] - y_smooth[-1-spacing]
                            time_diff = spacing / fps
                            
                            vx_improved = vx_diff / time_diff
                            vy_improved = vy_diff / time_diff
                            
                            # Calcular aceleraciones usando el mismo método
                            if len(data) >= 10:
                                # Obtener velocidades anteriores para calcular aceleración
                                df_vel = pd.DataFrame([d for d in data[-10:] if 'vx_m/s' in d])
                                if len(df_vel) >= 7:
                                    vx_data = df_vel['vx_m/s'].values
                                    vy_data = df_vel['vy_m/s'].values
                                    
                                    # Suavizar velocidades
                                    vx_smooth_data = smooth_data(vx_data)
                                    vy_smooth_data = smooth_data(vy_data)
                                    
                                    if len(vx_smooth_data) > spacing:
                                        ax_diff = vx_smooth_data[-1] - vx_smooth_data[-1-spacing]
                                        ay_diff = vy_smooth_data[-1] - vy_smooth_data[-1-spacing]
                                        
                                        smooth_acceleration_x = ax_diff / time_diff
                                        smooth_acceleration_y = ay_diff / time_diff
                            
                            # Usar valores mejorados como suavizados
                            smooth_velocity_x = vx_improved
                            smooth_velocity_y = vy_improved
                
                # Detectar velocidad inicial para predicción (cuando la velocidad Y es máxima)
                if not initial_velocity_detected and len(data) > 10:
                    recent_vy = [d['vy_m/s'] for d in data[-10:]]
                    if len(recent_vy) > 5 and velocity_y_m > 0:
                        # Si la velocidad Y actual es menor que las anteriores, hemos pasado el máximo
                        if all(velocity_y_m < vy for vy in recent_vy[-3:]):
                            initial_velocity_detected = True
                            initial_vx = velocity_x_m
                            initial_vy = max(recent_vy)
                            initial_pos_x = adjusted_x_m
                            initial_pos_y = adjusted_y_m
                            
                            # Calcular trayectoria predicha
                            pred_x, pred_y, pred_t = predict_trajectory(
                                initial_pos_x, initial_pos_y, initial_vx, initial_vy
                            )
                            predicted_trajectory_x = pred_x
                            predicted_trajectory_y = pred_y

            # Dibuja el rectángulo del objeto rastreado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              # Dibuja los vectores si están habilitados
            center_x = adjusted_x_px
            center_y = frame_height - adjusted_y_px
            
            # Vectores básicos (deshabilitados por defecto)
            if checkboxes["velocity"]["state"] and prev_x_m is not None:
                # Escala para visualización de vectores (ajustable)
                velocity_scale = 50  # píxeles por m/s
                
                # Calcula las coordenadas finales del vector de velocidad (usando ambas componentes)
                velocity_end_x = int(center_x + velocity_x_m * velocity_scale)
                velocity_end_y = int(center_y - velocity_y_m * velocity_scale)  # Negativo porque Y crece hacia abajo en OpenCV
                
                # Dibuja el vector de velocidad (flecha azul) - vector completo
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
                
                # Calcula las coordenadas finales del vector de aceleración
                accel_end_x = int(center_x + acceleration_x_m * acceleration_scale)
                accel_end_y = int(center_y - acceleration_y_m * acceleration_scale)
                
                # Dibuja el vector de aceleración (flecha roja)
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
                cv2.arrowedLine(frame, (center_x, center_y), vel_x_end, (0, 255, 255), 2, tipLength=0.2)                # Componente Y de velocidad (flecha amarilla)
                vel_y_end = (center_x, int(center_y - velocity_y_m * 50))
                cv2.arrowedLine(frame, (center_x, center_y), vel_y_end, (0, 255, 0), 2, tipLength=0.2)            # Dibuja vectores suavizados si están habilitados
            if checkboxes["smooth_vectors"]["state"] and len(data) > 7:
                velocity_scale_smooth = 50
                acceleration_scale_smooth = 10
                
                # Verificar si mostrar solo componentes Y
                if checkboxes["y_components"]["state"]:
                    # Solo componente Y de velocidade suavizada (flecha cyan vertical)
                    velocity_smooth_end_x = center_x  # Sin componente X
                    velocity_smooth_end_y = int(center_y - smooth_velocity_y * velocity_scale_smooth)
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (velocity_smooth_end_x, velocity_smooth_end_y),
                        (255, 255, 0),  # Cyan
                        3,
                        tipLength=0.3
                    )
                    
                    # Solo componente Y de aceleración suavizada (flecha magenta vertical)
                    accel_smooth_end_x = center_x  # Sin componente X
                    accel_smooth_end_y = int(center_y - smooth_acceleration_y * acceleration_scale_smooth)
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (accel_smooth_end_x, accel_smooth_end_y),
                        (255, 0, 255),  # Magenta
                        3,
                        tipLength=0.3
                    )
                else:
                    # Vector completo de velocidad suavizada (flecha cyan)
                    velocity_smooth_end_x = int(center_x + smooth_velocity_x * velocity_scale_smooth)
                    velocity_smooth_end_y = int(center_y - smooth_velocity_y * velocity_scale_smooth)
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (velocity_smooth_end_x, velocity_smooth_end_y),
                        (255, 255, 0),  # Cyan
                        3,
                        tipLength=0.3
                    )
                    
                    # Vector completo de aceleración suavizada (flecha magenta)
                    accel_smooth_end_x = int(center_x + smooth_acceleration_x * acceleration_scale_smooth)
                    accel_smooth_end_y = int(center_y - smooth_acceleration_y * acceleration_scale_smooth)
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (accel_smooth_end_x, accel_smooth_end_y),
                        (255, 0, 255),  # Magenta
                        3,
                        tipLength=0.3
                    )

            # Dibuja predicción de trayectoria si está habilitada
            if checkboxes["prediction"]["state"] and initial_velocity_detected and len(predicted_trajectory_x) > 0:
                # Convertir coordenadas predichas a píxeles
                for i in range(len(predicted_trajectory_x) - 1):
                    x1_pred = int((predicted_trajectory_x[i] / pixels_to_meters))
                    y1_pred = int(frame_height - ((predicted_trajectory_y[i] + offset_y_m) / pixels_to_meters))
                    x2_pred = int((predicted_trajectory_x[i+1] / pixels_to_meters))
                    y2_pred = int(frame_height - ((predicted_trajectory_y[i+1] + offset_y_m) / pixels_to_meters))
                    
                    # Verificar que los puntos estén dentro del frame
                    if (0 <= x1_pred < frame_width and 0 <= y1_pred < frame_height and
                        0 <= x2_pred < frame_width and 0 <= y2_pred < frame_height):
                        cv2.line(frame, (x1_pred, y1_pred), (x2_pred, y2_pred), (0, 165, 255), 2)  # Naranja            if checkboxes["magnitudes"]["state"]:
                cv2.putText(frame, f"X: {adjusted_x_m:.1f} m", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Y: {adjusted_y_m:.1f} m", (frame_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if prev_x_m is not None:
                    # Priorizar valores suavizados si están disponibles
                    if len(data) > 7:
                        if checkboxes["y_components"]["state"]:
                            # Mostrar solo componentes Y (suavizados)
                            cv2.putText(frame, f"Vy_suave: {smooth_velocity_y:.1f} m/s", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            cv2.putText(frame, f"Ay_suave: {smooth_acceleration_y:.1f} m/s^2", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        else:
                            # Mostrar valores suavizados completos
                            cv2.putText(frame, f"Vx_suave: {smooth_velocity_x:.1f} m/s", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(frame, f"Vy_suave: {smooth_velocity_y:.1f} m/s", (frame_width - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(frame, f"Ax_suave: {smooth_acceleration_x:.1f} m/s^2", (frame_width - 200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            cv2.putText(frame, f"Ay_suave: {smooth_acceleration_y:.1f} m/s^2", (frame_width - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    else:
                        # Valores básicos como respaldo
                        if checkboxes["y_components"]["state"]:
                            cv2.putText(frame, f"Vy: {velocity_y_m:.1f} m/s", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.putText(frame, f"Ay: {acceleration_y_m:.1f} m/s^2", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, f"Vx: {velocity_x_m:.1f} m/s", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.putText(frame, f"Vy: {velocity_y_m:.1f} m/s", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            cv2.putText(frame, f"Ax: {acceleration_x_m:.1f} m/s^2", (frame_width - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, f"Ay: {acceleration_y_m:.1f} m/s^2", (frame_width - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Información sobre detección de velocidad inicial
                    if initial_velocity_detected:
                        cv2.putText(frame, f"V0 detectada!", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if checkboxes["y_components"]["state"]:
                            cv2.putText(frame, f"V0y: {initial_vy:.1f} m/s", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"V0x: {initial_vx:.1f}, V0y: {initial_vy:.1f}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)           

           #Actualiza las variables previas
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

print("="*50)
print("ANÁLISIS MEJORADO")
print("="*50)

# PASO 1: Aplicar análisis mejorado con espaciado y suavizado
print("\n1. Aplicando método mejorado con espaciado y suavizado...")
df_improved = calculate_derivatives_with_spacing(df, spacing=3)

# PASO 2: Encontrar punto crítico (máximo de velocidad Y)
print("\n2. Encontrando punto crítico (máximo velocidad Y)...")
critical_frame, critical_idx = find_critical_time(df_improved)
if critical_frame is not None:
    print(f"   Punto crítico encontrado en frame: {critical_frame}")
    print(f"   Velocidad Y máxima: {df_improved.loc[critical_idx, 'vy_smooth']:.2f} m/s")
else:
    print("   No se pudo encontrar punto crítico")

# PASO 3: Análisis de caída libre para verificar gravedad
print("\n3. Analizando caída libre para estimar gravedad...")
gravity_params, gravity_estimate = analyze_free_fall(df_improved, critical_frame)
if gravity_estimate is not None:
    print(f"   Gravedad estimada: {gravity_estimate:.2f} m/s²")
    print(f"   Error respecto a 9.81 m/s²: {abs(gravity_estimate - 9.81):.2f} m/s²")
    print(f"   Error porcentual: {abs(gravity_estimate - 9.81)/9.81*100:.1f}%")
else:
    print("   No se pudo estimar la gravedad")

# PASO 4: Gráficos mejorados con datos suavizados
print("\n4. Generando gráficos mejorados...")
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Comparación posición: original vs suavizada
axs[0,0].plot(df["nro_frame"], df["x_m"], 'o-', alpha=0.5, label="X original", markersize=3)
axs[0,0].plot(df["nro_frame"], df["y_m"], 'o-', alpha=0.5, label="Y original", markersize=3)
if 'x_m_smooth' in df_improved.columns:
    axs[0,0].plot(df_improved["nro_frame"], df_improved["x_m_smooth"], '-', linewidth=2, label="X suavizada")
    axs[0,0].plot(df_improved["nro_frame"], df_improved["y_m_smooth"], '-', linewidth=2, label="Y suavizada")
axs[0,0].set_title("Posición: Original vs Suavizada")
axs[0,0].set_ylabel("Posición (m)")
axs[0,0].legend()
axs[0,0].grid(True)

# Comparación velocidad: original vs suavizada
axs[1,0].plot(df["nro_frame"], df["vx_m/s"], 'o-', alpha=0.5, label="Vx original", markersize=3)
axs[1,0].plot(df["nro_frame"], df["vy_m/s"], 'o-', alpha=0.5, label="Vy original", markersize=3)
if 'vx_smooth' in df_improved.columns:
    axs[1,0].plot(df_improved["nro_frame"], df_improved["vx_smooth"], '-', linewidth=2, label="Vx suavizada")
    axs[1,0].plot(df_improved["nro_frame"], df_improved["vy_smooth"], '-', linewidth=2, label="Vy suavizada")
axs[1,0].set_title("Velocidad: Original vs Suavizada")
axs[1,0].set_ylabel("Velocidad (m/s)")
axs[1,0].legend()
axs[1,0].grid(True)

# Comparación aceleración: original vs suavizada
axs[2,0].plot(df["nro_frame"], df["ax_m/s^2"], 'o-', alpha=0.5, label="Ax original", markersize=3)
axs[2,0].plot(df["nro_frame"], df["ay_m/s^2"], 'o-', alpha=0.5, label="Ay original", markersize=3)
if 'ax_calculated' in df_improved.columns:
    axs[2,0].plot(df_improved["nro_frame"], df_improved["ax_calculated"], '-', linewidth=2, label="Ax suavizada")
    axs[2,0].plot(df_improved["nro_frame"], df_improved["ay_calculated"], '-', linewidth=2, label="Ay suavizada")
axs[2,0].set_title("Aceleración: Original vs Suavizada")
axs[2,0].set_ylabel("Aceleración (m/s²)")
axs[2,0].set_xlabel("Frame")
axs[2,0].legend()
axs[2,0].grid(True)

# Trayectoria en el espacio
axs[0,1].plot(df["x_m"], df["y_m"], 'o-', alpha=0.7, label="Trayectoria original", markersize=4)
if critical_frame is not None:
    critical_x = df_improved.loc[critical_idx, 'x_m']
    critical_y = df_improved.loc[critical_idx, 'y_m']
    axs[0,1].plot(critical_x, critical_y, 'ro', markersize=10, label="Punto crítico")
axs[0,1].set_title("Trayectoria en el espacio")
axs[0,1].set_xlabel("X (m)")
axs[0,1].set_ylabel("Y (m)")
axs[0,1].legend()
axs[0,1].grid(True)
axs[0,1].axis('equal')

# Análisis de caída libre (solo después del punto crítico)
if critical_frame is not None:
    free_fall_data = df_improved[df_improved['nro_frame'] >= critical_frame]
    axs[1,1].plot(free_fall_data["nro_frame"], free_fall_data["vy_smooth"], 'o-', label="Vy durante caída libre")
    
    # Ajuste lineal
    if gravity_params is not None:
        x_fit = free_fall_data["nro_frame"].values
        y_fit = linear_function(x_fit, gravity_params[0], gravity_params[1])
        axs[1,1].plot(x_fit, y_fit, 'r--', linewidth=2, label=f"Ajuste lineal (pendiente={gravity_params[0]:.3f})")
    
    axs[1,1].set_title("Análisis de Caída Libre")
    axs[1,1].set_xlabel("Frame")
    axs[1,1].set_ylabel("Velocidad Y (m/s)")
    axs[1,1].legend()
    axs[1,1].grid(True)
else:
    axs[1,1].text(0.5, 0.5, "No se detectó\npunto crítico", ha='center', va='center', transform=axs[1,1].transAxes)
    axs[1,1].set_title("Análisis de Caída Libre")

# Verificación de la gravedad
axs[2,1].axhline(y=-9.81, color='r', linestyle='--', linewidth=2, label="Gravedad teórica (-9.81 m/s²)")
if 'ay_calculated' in df_improved.columns and critical_frame is not None:
    free_fall_accel = df_improved[df_improved['nro_frame'] >= critical_frame]["ay_calculated"]
    axs[2,1].plot(df_improved[df_improved['nro_frame'] >= critical_frame]["nro_frame"], 
                  free_fall_accel, 'o-', alpha=0.7, label="Aceleración Y medida")
    mean_accel = free_fall_accel.mean()
    axs[2,1].axhline(y=mean_accel, color='g', linestyle=':', linewidth=2, 
                     label=f"Promedio medido ({mean_accel:.2f} m/s²)")
axs[2,1].set_title("Verificación de Gravedad")
axs[2,1].set_xlabel("Frame")
axs[2,1].set_ylabel("Aceleración Y (m/s²)")
axs[2,1].legend()
axs[2,1].grid(True)

plt.tight_layout()
plt.savefig("analisis_mejorado.png", dpi=300, bbox_inches='tight')
plt.show()

# PASO 5: Guardar datos mejorados
print("\n5. Guardando datos mejorados...")
df_improved.to_csv("resultados_mejorados.csv", index=False)
print("   Datos mejorados guardados en 'resultados_mejorados.csv'")

# PASO 6: Resumen de resultados
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)
print(f"Frames totales analizados: {len(df)}")
print(f"Duración del experimento: {len(df)/60:.2f} segundos (asumiendo 60 FPS)")
if critical_frame is not None:
    print(f"Punto crítico (máximo Vy): Frame {critical_frame}")
    print(f"Tiempo hasta punto crítico: {critical_frame/60:.2f} segundos")
if gravity_estimate is not None:
    print(f"Gravedad estimada: {gravity_estimate:.2f} ± {abs(gravity_estimate - 9.81):.2f} m/s²")
    print(f"Precisión: {100 - abs(gravity_estimate - 9.81)/9.81*100:.1f}%")
print("="*50)
