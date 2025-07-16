import cv2
import numpy as np
import time
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suaviza una lista de datos usando un filtro Savitzky-Golay para sacar el ruido y ver mejor la tendencia.
def smooth_data(data, window_length=7, polyorder=2):
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def detect_initial_impulse_phase(df, threshold_acceleration=50):
    """
    Detecta la fase de impulso inicial basándose en aceleración alta
    Retorna el frame inicial y final del contacto humano
    """
    
    # Calcular magnitud de aceleración total
    df['accel_magnitude'] = np.sqrt(df['ax_m/s^2']**2 + df['ay_m/s^2']**2)
    
    # Suavizar para evitar picos espurios
    accel_smooth = smooth_data(df['accel_magnitude'].values, window_length=5)
    
    # Encontrar frames donde la aceleración supera el umbral
    high_accel_frames = df[accel_smooth > threshold_acceleration].index
    
    if len(high_accel_frames) == 0:
        return None, None
    
    # El impulso inicial debería estar al comienzo del movimiento
    # Tomar solo los primeros frames con alta aceleración
    contact_start = high_accel_frames[0] if len(high_accel_frames) > 0 else None
    
    # Encontrar cuando termina el contacto (cuando la aceleración vuelve a valores normales)
    contact_end = None
    for i in range(contact_start, min(contact_start + 20, len(df))):
        if accel_smooth[i] < threshold_acceleration:
            contact_end = i
            break
    
    if contact_end is None:
        contact_end = min(contact_start + 10, len(df) - 1)
    
    return contact_start, contact_end

def analyze_impulse_phase(df, contact_start, contact_end, ball_mass=0.458):
    #Analiza la fase de impulso inicial
    
    # Datos durante el contacto
    impulse_data = df.iloc[contact_start:contact_end+1].copy()
    
    if len(impulse_data) < 2:
        return None
    
    # Calcular parámetros del impulso
    fps = 60
    contact_time = len(impulse_data) / fps  # Tiempo de contacto en segundos

    # Velocidad inicial y final durante el contacto
    initial_velocity = 0  # Asumimos que la pelota parte del reposo
    final_velocity_x = impulse_data['vx_m/s'].iloc[-1]
    final_velocity_y = impulse_data['vy_m/s'].iloc[-1]
    final_velocity_magnitude = np.sqrt(final_velocity_x**2 + final_velocity_y**2)
    
    # Aceleración promedio durante el impulso
    avg_acceleration_x = impulse_data['ax_m/s^2'].mean()
    avg_acceleration_y = impulse_data['ay_m/s^2'].mean()
    avg_acceleration_magnitude = np.sqrt(avg_acceleration_x**2 + avg_acceleration_y**2)
    
    # Fuerza aplicada (F = ma)
    force_x = ball_mass * avg_acceleration_x
    force_y = ball_mass * avg_acceleration_y
    force_magnitude = ball_mass * avg_acceleration_magnitude
    
    # Impulso aplicado (I = F*t = m*Δv)
    impulse_x = ball_mass * final_velocity_x
    impulse_y = ball_mass * final_velocity_y
    impulse_magnitude = ball_mass * final_velocity_magnitude
    
    # Energía cinética inicial impartida
    kinetic_energy_initial = 0.5 * ball_mass * (final_velocity_magnitude**2)
    #kinetic_energy_initial = 0

    return {
        'contact_start_frame': contact_start,
        'contact_end_frame': contact_end,
        'contact_time_s': contact_time,
        'final_velocity_x_ms': final_velocity_x,
        'final_velocity_y_ms': final_velocity_y,
        'final_velocity_magnitude_ms': final_velocity_magnitude,
        'avg_acceleration_x_ms2': avg_acceleration_x,
        'avg_acceleration_y_ms2': avg_acceleration_y,
        'avg_acceleration_magnitude_ms2': avg_acceleration_magnitude,
        'force_x_N': force_x,
        'force_y_N': force_y,
        'force_magnitude_N': force_magnitude,
        'impulse_x_Ns': impulse_x,
        'impulse_y_Ns': impulse_y,
        'impulse_magnitude_Ns': impulse_magnitude,
        'kinetic_energy_initial_J': kinetic_energy_initial,
        'ball_mass_kg': ball_mass
    }

#Primero suaviza las posiciones, luego se deriva para obtener velocidad y aceleración, aplicando suavizado en cada paso para reducir el ruido.
def calculate_derivatives_with_spacing(df, spacing=3):
    # DATAFRAME SUAVIZADO
    df_smooth = df.copy()
    
    # Suavizar posición
    df_smooth['x_m_smooth'] = smooth_data(df['x_m'].values)
    df_smooth['y_m_smooth'] = smooth_data(df['y_m'].values)
    
    # VELOCIDAD CON ESPACIADO 
    df_smooth['vx_calculated'] = df_smooth['x_m_smooth'].diff(spacing) / (spacing / 60)  #  
    df_smooth['vy_calculated'] = df_smooth['y_m_smooth'].diff(spacing) / (spacing / 60)
    
    # Suavizar velocidad
    df_smooth['vx_smooth'] = smooth_data(df_smooth['vx_calculated'].fillna(0).values)
    df_smooth['vy_smooth'] = smooth_data(df_smooth['vy_calculated'].fillna(0).values)
    
    # Calcular aceleracion con espaciado
    df_smooth['ax_calculated'] = df_smooth['vx_smooth'].diff(spacing) / (spacing / 60)
    df_smooth['ay_calculated'] = df_smooth['vy_smooth'].diff(spacing) / (spacing / 60)
    
    return df_smooth

#Encuentra el tiempo critico donde la velocidad en Y es max
def find_critical_time(df):
    if 'vy_smooth' in df.columns:
        max_idx = df['vy_smooth'].idxmax()
        return df.loc[max_idx, 'nro_frame'], max_idx
    return None, None

def linear_function(x, a, b):
    #funcion lineal ajuste = y= ax + b
    return a * x + b

#analizamos caida libre después del punto crítico y antes del rebote
def analyze_free_fall(df, critical_frame):
    if critical_frame is None:
        return None, None
    
    # Datos después del punto crítico
    free_fall_data = df[df['nro_frame'] >= critical_frame].copy()
    
    if len(free_fall_data) < 5:
        return None, None
    
    # DETECTAR Y EXCLUIR LA FASE DE REBOTE
    # Buscar el punto donde la pelota toca el suelo (altura min)
    min_height_idx = free_fall_data['y_m'].idxmin()
    min_height_frame = free_fall_data.loc[min_height_idx, 'nro_frame']
    
    # Usar solo los datos ANTES del rebote (caida libre pura)
    # Añadir un margen de seguridad de 5 frames antes del impacto
    safety_margin = 5
    impact_frame = min_height_frame - safety_margin
    
    # Filtrar datos solo hasta el impacto (excluyendo rebote)
    pure_free_fall = free_fall_data[free_fall_data['nro_frame'] <= impact_frame].copy()
    
    print(f"    Análisis de caída libre:")
    print(f"      - Punto crítico (máximo Y): Frame {critical_frame}")
    print(f"      - Impacto detectado en: Frame {min_height_frame}")
    print(f"      - Usando frames {critical_frame} a {impact_frame} (caída libre pura)")
    print(f"      - Excluyendo {len(free_fall_data) - len(pure_free_fall)} frames de rebote")
    
    # Ajustar linea recta a la velocidad en Y durante caída libre pura
    x_data = pure_free_fall['nro_frame'].values
    y_data = pure_free_fall['vy_smooth'].values
    
    try:
        popt, pcov = curve_fit(linear_function, x_data, y_data)
        gravity_estimate = abs(popt[0]) * 60  # Convertir a m/s² considerando FPS
        return popt, gravity_estimate
    except:
        return None, None

#predecir trayectoria
def predict_trajectory(x0, y0, vx0, vy0, t_max=2.0, dt=0.01, g=9.81):
    t = np.arange(0, t_max, dt)
    x_pred = x0 + vx0 * t
    y_pred = y0 + vy0 * t - 0.5 * g * t**2
    
    # solo positivo en y
    valid_idx = y_pred >= 0
    return x_pred[valid_idx], y_pred[valid_idx], t[valid_idx]

# diametro pelota en metros
ball_diameter_m = 0.24

# variables figura
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
    "prediction": {"pos": (10, 120), "size": (20, 20), "label": "Prediccion", "state": True},
    "smooth_vectors": {"pos": (10, 150), "size": (20, 20), "label": "Vectores Suavizados", "state": True},
    "y_components": {"pos": (10, 180), "size": (20, 20), "label": "Solo Componentes Y", "state": True},
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

# hardcodeada del objeto a rastrear
def draw_circle(event, x, y, flags, param):
    pass

# Valores hardcodeados
bbox = (297 - 25, 620 - 25, 50, 50)  # (x, y, w, h)
pixels_to_meters = 0.004800  # m/px

# Inicializa la captura de video
cap = cv2.VideoCapture("video/video.mp4")

ret, frame = cap.read()

# relacion pixel-metro
pixels_diameter = bbox[2]  # Ancho del rectangulo en píxeles
pixels_to_meters = ball_diameter_m / pixels_diameter

tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# dimensiones del video
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# variables para calcular velocidad y aceleracion
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

# variables para suavizado y prediccion
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

paused = False

# datos dataframe
data = []
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

            # Mostrar cronómetro en el video
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            milliseconds = int((current_time * 1000) % 1000)
            time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

            cv2.putText(frame, f"Tiempo: {time_str}", (10, frame_height - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
                        cv2.putText(frame, f"V0 detectada!", (10, frame_height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        if checkboxes["y_components"]["state"]:
                            cv2.putText(frame, f"V0y: {initial_vy:.1f} m/s", (10, frame_height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"V0x: {initial_vx:.1f}, V0y: {initial_vy:.1f}", (10, frame_height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)           

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

# =============================
# CÁLCULO DE ENERGÍAS
# =============================

ball_mass = 0.486  # kg (ya lo usás antes)
g = 9.81  # m/s²

# Asegurarse de que existen columnas necesarias
if "vx_m/s" in df.columns and "vy_m/s" in df.columns and "y_m" in df.columns:
    # Calcular velocidad total
    df["v_total"] = np.sqrt(df["vx_m/s"]**2 + df["vy_m/s"]**2)

    # Energía cinética
    df["E_cinetica"] = 0.5 * ball_mass * df["v_total"]**2

    # Energía potencial (altura respecto al suelo)
    df["E_potencial"] = ball_mass * g * (df["y_m"] + 1)  # +1 por el offset_y_m

    # Energía mecánica total
    df["E_mecanica"] = df["E_cinetica"] + df["E_potencial"]

print("="*50)
print("ANÁLISIS MEJORADO")
print("="*50)

# PASO 1: Aplicar análisis mejorado con espaciado y suavizado
print("\n1. Aplicando método mejorado con espaciado y suavizado...")
df_improved = calculate_derivatives_with_spacing(df, spacing=3)

# PASO 1.5: ANÁLISIS DE FASE 1 - IMPULSO INICIAL
print("\n1.5. FASE 1: Analizando impulso inicial (contacto humano)...")
contact_start, contact_end = detect_initial_impulse_phase(df)
impulse_analysis = analyze_impulse_phase(df, contact_start, contact_end)

if impulse_analysis is not None:
    print(f"   ✅ Fase de contacto detectada:")
    print(f"      - Frames de contacto: {impulse_analysis['contact_start_frame']} a {impulse_analysis['contact_end_frame']}")
    print(f"      - Tiempo de contacto: {impulse_analysis['contact_time_s']:.3f} segundos")
    print(f"      - Velocidad final impartida: {impulse_analysis['final_velocity_magnitude_ms']:.2f} m/s")
    print(f"      - Aceleración promedio: {impulse_analysis['avg_acceleration_magnitude_ms2']:.1f} m/s²")
    print(f"      - Fuerza aplicada: {impulse_analysis['force_magnitude_N']:.1f} N")
    print(f"      - Impulso total: {impulse_analysis['impulse_magnitude_Ns']:.2f} N·s")
    print(f"      - Energía cinética inicial: {impulse_analysis['kinetic_energy_initial_J']:.1f} J")
else:
    print("   ❌ No se pudo detectar la fase de impulso inicial")
    print("       Posibles causas:")
    print("       - El video no captura el momento de contacto")
    print("       - La aceleración durante el contacto es muy baja")
    print("       - Se necesita ajustar el umbral de detección")

# PASO 2: Encontrar punto crítico (máximo de velocidad Y)
print("\n2. FASE 2: Encontrando punto crítico (máximo velocidad Y)...")
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

# PASO 4: Gráficos mejorados con datos suavizados usando Plotly
print("\n4. Generando gráficos interactivos completos con ambas fases...")

# Crear figura de energías
energia_fig = go.Figure()

energia_fig.add_trace(go.Scatter(
    x=df["nro_frame"], y=df["E_cinetica"],
    mode='lines', name="Energía Cinética", line=dict(color="red", width=3)
))

energia_fig.add_trace(go.Scatter(
    x=df["nro_frame"], y=df["E_potencial"],
    mode='lines', name="Energía Potencial", line=dict(color="blue", width=3)
))

energia_fig.add_trace(go.Scatter(
    x=df["nro_frame"], y=df["E_mecanica"],
    mode='lines', name="Energía Mecánica Total", line=dict(color="green", width=3, dash='dash')
))

energia_fig.update_layout(
    title="Energías durante el movimiento",
    xaxis_title="Frame",
    yaxis_title="Energía (J)",
    legend=dict(x=0.01, y=0.99, bgcolor="white", bordercolor="black"),
    height=500,
    width=900
)

energia_fig.write_html("grafico_energias.html")
print("   ✅ Gráfico de energías guardado como 'grafico_energias.html'")

# Crear subplots con Plotly - Mejorado con más espacio y mejor diseño
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        "<b>Posición: Original vs Suavizada</b>", 
        "<b>Trayectoria en el espacio</b>", 
        "<b>FASE 1: Aceleración durante Impulso</b>",
        "<b>Velocidad: Original vs Suavizada</b>", 
        "<b>Análisis de Caída Libre</b>", 
        "<b>FASE 1: Velocidad durante Impulso</b>", 
        "<b>Aceleración: Original vs Suavizada</b>", 
        "<b>Verificación de Gravedad</b>", 
        "<b>FASE 1: Resumen de Parámetros</b>"
    ],
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"type": "bar"}]],
    vertical_spacing=0.12,  # Aumentado para mejor separación
    horizontal_spacing=0.10  # Aumentado para mejor separación
)

# (1,1) Comparación posición: original vs suavizada
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["x_m"], 
               mode='markers+lines', name="X original", 
               opacity=0.6, marker=dict(size=4, color='blue'), 
               line=dict(width=2, color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["y_m"], 
               mode='markers+lines', name="Y original", 
               opacity=0.6, marker=dict(size=4, color='red'), 
               line=dict(width=2, color='red')),
    row=1, col=1
)

if 'x_m_smooth' in df_improved.columns:
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["x_m_smooth"], 
                   mode='lines', name="X suavizada", 
                   line=dict(width=4, color='darkblue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["y_m_smooth"], 
                   mode='lines', name="Y suavizada", 
                   line=dict(width=4, color='darkred')),
        row=1, col=1
    )

# (2,1) Comparación velocidad: original vs suavizada
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["vx_m/s"], 
               mode='markers+lines', name="Vx original", 
               opacity=0.6, marker=dict(size=4, color='green'), 
               line=dict(width=2, color='green')),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["vy_m/s"], 
               mode='markers+lines', name="Vy original", 
               opacity=0.6, marker=dict(size=4, color='orange'), 
               line=dict(width=2, color='orange')),
    row=2, col=1
)

if 'vx_smooth' in df_improved.columns:
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["vx_smooth"], 
                   mode='lines', name="Vx suavizada", 
                   line=dict(width=4, color='darkgreen')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["vy_smooth"], 
                   mode='lines', name="Vy suavizada", 
                   line=dict(width=4, color='darkorange')),
        row=2, col=1
    )

# (3,1) Comparación aceleración: original vs suavizada
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["ax_m/s^2"], 
               mode='markers+lines', name="Ax original", 
               opacity=0.6, marker=dict(size=4, color='purple'), 
               line=dict(width=2, color='purple')),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=df["nro_frame"], y=df["ay_m/s^2"], 
               mode='markers+lines', name="Ay original", 
               opacity=0.6, marker=dict(size=4, color='brown'), 
               line=dict(width=2, color='brown')),
    row=3, col=1
)

if 'ax_calculated' in df_improved.columns:
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["ax_calculated"], 
                   mode='lines', name="Ax suavizada", 
                   line=dict(width=4, color='darkviolet')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_improved["nro_frame"], y=df_improved["ay_calculated"], 
                   mode='lines', name="Ay suavizada", 
                   line=dict(width=4, color='saddlebrown')),
        row=3, col=1
    )

# (1,2) Trayectoria en el espacio
fig.add_trace(
    go.Scatter(x=df["x_m"], y=df["y_m"], 
               mode='markers+lines', name="Trayectoria original", 
               opacity=0.8, marker=dict(size=6, color='blue'), 
               line=dict(width=3, color='blue')),
    row=1, col=2
)

if critical_frame is not None:
    critical_x = df_improved.loc[critical_idx, 'x_m']
    critical_y = df_improved.loc[critical_idx, 'y_m']
    fig.add_trace(
        go.Scatter(x=[critical_x], y=[critical_y], 
                   mode='markers', name="Punto crítico", 
                   marker=dict(size=15, color='red', symbol='star', 
                              line=dict(width=2, color='darkred'))),
        row=1, col=2
    )

# (2,2) Análisis de caída libre
if critical_frame is not None:
    free_fall_data = df_improved[df_improved['nro_frame'] >= critical_frame]
    fig.add_trace(
        go.Scatter(x=free_fall_data["nro_frame"], y=free_fall_data["vy_smooth"], 
                   mode='markers+lines', name="Vy durante caída libre",
                   marker=dict(size=6, color='blue'),
                   line=dict(width=3, color='blue')),
        row=2, col=2
    )
    
    # Ajuste lineal
    if gravity_params is not None:
        x_fit = free_fall_data["nro_frame"].values
        y_fit = linear_function(x_fit, gravity_params[0], gravity_params[1])
        fig.add_trace(
            go.Scatter(x=x_fit, y=y_fit, 
                       mode='lines', name=f"Ajuste lineal (pendiente={gravity_params[0]:.3f})",
                       line=dict(dash='dash', color='red', width=4)),
            row=2, col=2
        )
else:
    # Agregar texto cuando no hay punto crítico
    fig.add_annotation(
        text="<b>No se detectó<br>punto crítico</b>",
        x=0.5, y=0.5, xref=f"x{4}", yref=f"y{4}",
        showarrow=False, font=dict(size=16, color='red')
    )

# (3,2) Verificación de la gravedad
fig.add_hline(y=-9.81, line_dash="dash", line_color="red", line_width=3,
              annotation_text="Gravedad teórica (-9.81 m/s²)",
              annotation_position="top right",
              row=3, col=2)

if 'ay_calculated' in df_improved.columns and critical_frame is not None:
    # EXCLUIR LA FASE DE REBOTE DEL CÁLCULO DE GRAVEDAD
    # Detectar el punto de impacto (altura mínima)
    free_fall_full = df_improved[df_improved['nro_frame'] >= critical_frame]
    min_height_idx = free_fall_full['y_m'].idxmin()
    min_height_frame = free_fall_full.loc[min_height_idx, 'nro_frame']
    
    # Usar solo datos ANTES del rebote con margen de seguridad
    safety_margin = 5
    impact_frame = min_height_frame - safety_margin
    
    # Filtrar datos de caída libre pura (sin rebote)
    pure_free_fall_accel = df_improved[
        (df_improved['nro_frame'] >= critical_frame) & 
        (df_improved['nro_frame'] <= impact_frame)
    ]
    
    if len(pure_free_fall_accel) > 0:
        # Mostrar todos los datos de caída libre en gris claro
        fig.add_trace(
            go.Scatter(x=df_improved[df_improved['nro_frame'] >= critical_frame]["nro_frame"], 
                       y=df_improved[df_improved['nro_frame'] >= critical_frame]["ay_calculated"], 
                       mode='markers+lines', name="Todos los datos (con rebote)",
                       opacity=0.4, marker=dict(size=4, color='lightgray'),
                       line=dict(width=2, color='lightgray')),
            row=3, col=2
        )
        
        # Resaltar solo los datos de caída libre pura
        fig.add_trace(
            go.Scatter(x=pure_free_fall_accel["nro_frame"], 
                       y=pure_free_fall_accel["ay_calculated"], 
                       mode='markers+lines', name="Caída libre pura (sin rebote)",
                       opacity=0.9, marker=dict(size=6, color='blue'),
                       line=dict(width=4, color='blue')),
            row=3, col=2
        )
        
        # Calcular promedio solo de caída libre pura
        mean_accel_pure = pure_free_fall_accel["ay_calculated"].mean()
        fig.add_hline(y=mean_accel_pure, line_dash="dot", line_color="green", line_width=4,
                      annotation_text=f"Promedio sin rebote ({mean_accel_pure:.2f} m/s²)",
                      annotation_position="bottom right",
                      row=3, col=2)
        
        # Agregar línea vertical indicando el punto de impacto
        fig.add_vline(x=min_height_frame, line_dash="dashdot", line_color="orange", line_width=2,
                      annotation_text="Impacto detectado",
                      annotation_position="top",
                      row=3, col=2)
    else:
        # Fallback a datos completos si no hay suficientes datos
        free_fall_accel = df_improved[df_improved['nro_frame'] >= critical_frame]["ay_calculated"]
        fig.add_trace(
            go.Scatter(x=df_improved[df_improved['nro_frame'] >= critical_frame]["nro_frame"], 
                       y=free_fall_accel, 
                       mode='markers+lines', name="Aceleración Y medida",
                       opacity=0.8, marker=dict(size=5, color='blue'),
                       line=dict(width=3, color='blue')),
            row=3, col=2
        )
        mean_accel_pure = free_fall_accel.mean()
        fig.add_hline(y=mean_accel_pure, line_dash="dot", line_color="green", line_width=3,
                      annotation_text=f"Promedio medido ({mean_accel_pure:.2f} m/s²)",
                      annotation_position="bottom right",
                      row=3, col=2)

# GRÁFICOS PARA FASE 1 (Columna 3)
if impulse_analysis is not None:
    # (1,3) Fase de impulso - Aceleración vs tiempo
    impulse_data_frames = df.iloc[impulse_analysis['contact_start_frame']:
                                 impulse_analysis['contact_end_frame'] + 1]
    
    fig.add_trace(
        go.Scatter(x=impulse_data_frames["nro_frame"], y=impulse_data_frames["ax_m/s^2"], 
                   mode='markers+lines', name="Aceleración X (Impulso)",
                   marker=dict(size=7, color='red'),
                   line=dict(color='red', width=4)),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=impulse_data_frames["nro_frame"], y=impulse_data_frames["ay_m/s^2"], 
                   mode='markers+lines', name="Aceleración Y (Impulso)",
                   marker=dict(size=7, color='blue'),
                   line=dict(color='blue', width=4)),
        row=1, col=3
    )
    
    # Líneas promedio con mejor estilo
    fig.add_hline(y=impulse_analysis['avg_acceleration_x_ms2'], 
                  line_dash="dash", line_color="red", line_width=2, opacity=0.8,
                  annotation_text=f"Ax promedio: {impulse_analysis['avg_acceleration_x_ms2']:.1f} m/s²",
                  annotation_position="top left",
                  row=1, col=3)
    fig.add_hline(y=impulse_analysis['avg_acceleration_y_ms2'], 
                  line_dash="dash", line_color="blue", line_width=2, opacity=0.8,
                  annotation_text=f"Ay promedio: {impulse_analysis['avg_acceleration_y_ms2']:.1f} m/s²",
                  annotation_position="bottom left",
                  row=1, col=3)
    
    # (2,3) Fase de impulso - Velocidad vs tiempo
    fig.add_trace(
        go.Scatter(x=impulse_data_frames["nro_frame"], y=impulse_data_frames["vx_m/s"], 
                   mode='markers+lines', name="Velocidad X (Impulso)",
                   marker=dict(size=7, color='red'),
                   line=dict(color='red', width=4)),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(x=impulse_data_frames["nro_frame"], y=impulse_data_frames["vy_m/s"], 
                   mode='markers+lines', name="Velocidad Y (Impulso)",
                   marker=dict(size=7, color='blue'),
                   line=dict(color='blue', width=4)),
        row=2, col=3
    )
    
    # Velocidad final con mejor estilo
    fig.add_hline(y=impulse_analysis['final_velocity_x_ms'], 
                  line_dash="dot", line_color="red", line_width=2, opacity=0.8,
                  annotation_text=f"Vx final: {impulse_analysis['final_velocity_x_ms']:.1f} m/s",
                  annotation_position="top right",
                  row=2, col=3)
    fig.add_hline(y=impulse_analysis['final_velocity_y_ms'], 
                  line_dash="dot", line_color="blue", line_width=2, opacity=0.8,
                  annotation_text=f"Vy final: {impulse_analysis['final_velocity_y_ms']:.1f} m/s",
                  annotation_position="bottom right",
                  row=2, col=3)
    
    # (3,3) Resumen de fuerzas y energía - Mejorado
    categories = ['Fuerza X<br>(N)', 'Fuerza Y<br>(N)', 'Fuerza Total<br>(N)', 
                  'Energía Inicial<br>(J)', 'Impulso Total<br>(N·s)']
    values = [abs(impulse_analysis['force_x_N']), abs(impulse_analysis['force_y_N']), 
              impulse_analysis['force_magnitude_N'], impulse_analysis['kinetic_energy_initial_J'],
              impulse_analysis['impulse_magnitude_Ns']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig.add_trace(
        go.Bar(x=categories, y=values, 
               marker=dict(color=colors, line=dict(width=2, color='black')),
               text=[f'<b>{v:.1f}</b>' for v in values],
               textposition='outside',
               textfont=dict(size=12, color='black'),
               name="Parámetros Fase 1"),
        row=3, col=3
    )
else:
    # Si no se detectó impulso, agregar texto en los gráficos de fase 1
    for i, col in enumerate([1, 2, 3], 1):
        fig.add_annotation(
            text="<b>FASE 1: Impulso<br>no detectado</b><br><br>El video podría no<br>capturar el momento<br>de contacto inicial",
            x=0.5, y=0.5, xref=f"x{i+6}", yref=f"y{i+6}",
            showarrow=False, font=dict(size=14, color='orange'),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="orange", borderwidth=2
        )

# Actualizar layout - Mejorado con mayor tamaño y mejor diseño
fig.update_layout(
    height=1400,  # Aumentado significativamente
    width=2000,   # Aumentado significativamente
    title_text="<b>Análisis Completo de Dinámica del Proyectil - Ambas Fases</b>",
    title_x=0.5,
    title_font_size=20,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.01,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=11),
    margin=dict(l=80, r=200, t=100, b=80)
)

# Actualizar ejes con mejor formato y grids
axis_config = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    showline=True,
    linewidth=2,
    linecolor='black',
    mirror=True,
    tickfont=dict(size=11)
)

# Aplicar configuración a todos los ejes X
for i in range(1, 10):
    fig.update_xaxes(axis_config, row=(i-1)//3+1, col=(i-1)%3+1)

# Aplicar configuración a todos los ejes Y  
for i in range(1, 10):
    fig.update_yaxes(axis_config, row=(i-1)//3+1, col=(i-1)%3+1)

# Títulos específicos de ejes con mejor formato
fig.update_xaxes(title_text="<b>Frame</b>", row=3, col=1)
fig.update_xaxes(title_text="<b>X (m)</b>", row=1, col=2)
fig.update_xaxes(title_text="<b>Frame</b>", row=2, col=2)
fig.update_xaxes(title_text="<b>Frame</b>", row=3, col=2)
fig.update_xaxes(title_text="<b>Frame</b>", row=1, col=3)
fig.update_xaxes(title_text="<b>Frame</b>", row=2, col=3)
fig.update_xaxes(title_text="<b>Parámetros</b>", row=3, col=3)

fig.update_yaxes(title_text="<b>Posición (m)</b>", row=1, col=1)
fig.update_yaxes(title_text="<b>Velocidad (m/s)</b>", row=2, col=1)
fig.update_yaxes(title_text="<b>Aceleración (m/s²)</b>", row=3, col=1)
fig.update_yaxes(title_text="<b>Y (m)</b>", row=1, col=2)
fig.update_yaxes(title_text="<b>Velocidad Y (m/s)</b>", row=2, col=2)
fig.update_yaxes(title_text="<b>Aceleración Y (m/s²)</b>", row=3, col=2)
fig.update_yaxes(title_text="<b>Aceleración (m/s²)</b>", row=1, col=3)
fig.update_yaxes(title_text="<b>Velocidad (m/s)</b>", row=2, col=3)
fig.update_yaxes(title_text="<b>Magnitud</b>", row=3, col=3)

# Mejorar aspecto del gráfico de trayectoria (mantener proporciones)
fig.update_xaxes(scaleanchor="y1", scaleratio=1, row=1, col=2)
fig.update_yaxes(scaleanchor="x1", scaleratio=1, row=1, col=2)

# Guardar y mostrar con mejor calidad
fig.write_html("analisis_mejorado_interactivo.html")
print("   ✅ Archivo HTML interactivo guardado: 'analisis_mejorado_interactivo.html'")

# Mostrar en el navegador
try:
    fig.show()
    print("   ✅ Gráficos abiertos en el navegador")
except Exception as e:
    print(f"   ⚠️ No se pudo abrir automáticamente en el navegador: {e}")
    print("   💡 Abra manualmente el archivo 'analisis_mejorado_interactivo.html'")

# PASO 5: Guardar datos mejorados
print("\n5. Guardando datos mejorados...")
df_improved.to_csv("resultados_mejorados.csv", index=False)
print("   ✅ Datos mejorados guardados en 'resultados_mejorados.csv'")

# PASO 6: Resumen completo de ambas fases
print("\n" + "="*50)
print("RESUMEN COMPLETO - ANÁLISIS DE AMBAS FASES")
print("="*50)

print(f"📊 DATOS GENERALES:")
print(f"   • Frames totales analizados: {len(df)}")
print(f"   • Duración del experimento: {len(df)/60:.2f} segundos (asumiendo 60 FPS)")

print(f"\n🚀 FASE 1 - IMPULSO INICIAL (Contacto Humano):")
if impulse_analysis is not None:
    print(f"   ✅ Impulso detectado exitosamente")
    print(f"   • Tiempo de contacto: {impulse_analysis['contact_time_s']:.3f} segundos")
    print(f"   • Velocidad inicial impartida: {impulse_analysis['final_velocity_magnitude_ms']:.2f} m/s")
    print(f"   • Aceleración durante contacto: {impulse_analysis['avg_acceleration_magnitude_ms2']:.1f} m/s²")
    print(f"   • Fuerza aplicada: {impulse_analysis['force_magnitude_N']:.1f} N")
    print(f"   • Energía cinética inicial: {impulse_analysis['kinetic_energy_initial_J']:.1f} J")
    print(f"   • Impulso total aplicado: {impulse_analysis['impulse_magnitude_Ns']:.2f} N·s")
else:
    print(f"   ❌ Impulso NO detectado")
    print(f"   • El video posiblemente no captura el momento de contacto")
    print(f"   • Se recomienda grabar desde antes del lanzamiento")

print(f"\n🎯 FASE 2 - VUELO LIBRE:")
if critical_frame is not None:
    print(f"   ✅ Análisis de vuelo libre completado")
    print(f"   • Punto crítico (máximo Vy): Frame {critical_frame}")
    print(f"   • Tiempo hasta punto crítico: {critical_frame/60:.2f} segundos")
    print(f"   • Velocidad Y máxima: {df_improved.loc[critical_idx, 'vy_smooth']:.2f} m/s")
else:
    print(f"   ⚠️ Punto crítico no encontrado claramente")

if gravity_estimate is not None:
    print(f"\n🌍 VALIDACIÓN FÍSICA:")
    print(f"   • Gravedad estimada: {gravity_estimate:.2f} m/s²")
    print(f"   • Error absoluto: ±{abs(gravity_estimate - 9.81):.2f} m/s²")
    print(f"   • Precisión del experimento: {100 - abs(gravity_estimate - 9.81)/9.81*100:.1f}%")

print(f"\n📈 COMPARACIÓN DE FASES:")
if impulse_analysis is not None and gravity_estimate is not None:
    accel_ratio = impulse_analysis['avg_acceleration_magnitude_ms2'] / 9.81
    print(f"   • Aceleración impulso vs gravedad: {accel_ratio:.1f}x")
    print(f"   • Duración impulso vs vuelo libre: {impulse_analysis['contact_time_s']:.3f}s vs {len(df)/60:.2f}s")

print("="*50)

