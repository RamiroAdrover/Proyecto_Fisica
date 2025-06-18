import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Importar las funciones de análisis del archivo principal
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

def load_selection_data():
    """Carga los datos de selección precisa o ejecuta el selector"""
    selection_file = "seleccion_precisa.json"
    
    if os.path.exists(selection_file):
        print(f"Cargando selección desde {selection_file}...")
        with open(selection_file, 'r') as f:
            data = json.load(f)
        
        print(f"Datos cargados:")
        print(f"  Centro: {data['center']}")
        print(f"  Radio: {data['radius']}")
        print(f"  BBox: {data['bbox']}")
        print(f"  Diámetro: {data['pixels_diameter']} píxeles")
        
        return data
    else:
        print(f"No se encontró {selection_file}")
        print("Ejecutando selector de objeto...")
        
        # Ejecutar selector
        from selector_objeto import CircleSelector
        selector = CircleSelector("video/video.mp4")
        result = selector.run()
        
        if result is not None:
            # Guardar resultado
            with open(selection_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Selección guardada en {selection_file}")
            return result
        else:
            print("Error: No se pudo obtener selección")
            return None

# Función de callback para manejar clics del mouse
def mouse_callback(event, x, y, flags, param):
    global checkboxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for key, checkbox in checkboxes.items():
            cx, cy = checkbox["pos"]
            cw, ch = checkbox["size"]
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                checkbox["state"] = not checkbox["state"]

def main():
    # Cargar datos de selección precisa
    selection_data = load_selection_data()
    if selection_data is None:
        print("Error: No se pudo obtener la selección del objeto")
        return
    
    # Extraer datos precisos
    precise_center = selection_data['center']
    precise_radius = selection_data['radius']
    precise_bbox = selection_data['bbox']
    pixels_diameter = selection_data['pixels_diameter']
    
    # Diámetro de la pelota en metros
    ball_diameter_m = 0.24
    pixels_to_meters = ball_diameter_m / pixels_diameter
    
    print(f"\nUsando selección precisa:")
    print(f"Centro: {precise_center}")
    print(f"Radio: {precise_radius} píxeles")
    print(f"Diámetro: {pixels_diameter} píxeles")
    print(f"Relación píxeles a metros: {pixels_to_meters:.6f} m/px")
    print("="*50)
    
    # Variables globales para los checkboxes
    global checkboxes
    checkboxes = {
        "velocity": {"pos": (10, 30), "size": (20, 20), "label": "Velocidad Basica", "state": False},
        "acceleration": {"pos": (10, 60), "size": (20, 20), "label": "Aceleracion Basica", "state": False},
        "magnitudes": {"pos": (10, 90), "size": (20, 20), "label": "Magnitudes", "state": True},
        "prediction": {"pos": (10, 120), "size": (20, 20), "label": "Prediccion", "state": True},
        "smooth_vectors": {"pos": (10, 150), "size": (20, 20), "label": "Vectores Suavizados", "state": True},
        "y_components": {"pos": (10, 180), "size": (20, 20), "label": "Solo Componentes Y", "state": True},
        "circle_overlay": {"pos": (10, 210), "size": (20, 20), "label": "Mostrar Circulo Preciso", "state": True},
    }
    
    # Inicializar captura de video
    cap = cv2.VideoCapture("video/video.mp4")
    
    # Configurar tracker con selección precisa
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return
    
    # Inicializar tracker CSRT con bbox precisa
    tracker = cv2.TrackerCSRT_create()
    success = tracker.init(frame, precise_bbox)
    
    if not success:
        print("Error: No se pudo inicializar el tracker")
        return
    
    print("Tracker inicializado exitosamente con selección precisa")
    
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
    cv2.namedWindow("Rastreo CSRT - Selección Precisa")
    cv2.setMouseCallback("Rastreo CSRT - Selección Precisa", mouse_callback)
    
    print("\nIniciando rastreo...")
    print("Controles:")
    print("- P: Pausar/Reanudar")
    print("- Q: Salir")
    print("- Clic en checkboxes para cambiar visualización")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("Fin del video o error al leer el frame.")
                break
            
            # Actualizar el tracker
            success, bbox = tracker.update(frame)
            
            if success:
                # Coordenadas del objeto rastreado
                x, y, w, h = [int(v) for v in bbox]
                
                # Ajustar el sistema de coordenadas: origen en la esquina inferior izquierda
                adjusted_y_px = frame_height - (y + h // 2)
                adjusted_x_px = x + w // 2
                
                # Convertir las coordenadas a metros usando la relación precisa
                adjusted_y_m = (adjusted_y_px * pixels_to_meters) - offset_y_m 
                adjusted_x_m = adjusted_x_px * pixels_to_meters
                
                # Calcular el tiempo actual
                current_time = 1 / cap.get(cv2.CAP_PROP_FPS) * frame_count
                
                if prev_x_m is not None and prev_y_m is not None and prev_time is not None:
                    # Calcular la velocidad en ambos ejes (m/s) - método básico
                    delta_time = current_time - prev_time
                    velocity_x_m = (adjusted_x_m - prev_x_m) / delta_time
                    velocity_y_m = (adjusted_y_m - prev_y_m) / delta_time
                    
                    # Calcular la aceleración en ambos ejes (m/s²) - método básico
                    acceleration_x_m = (velocity_x_m - prev_velocity_x_m) / delta_time
                    acceleration_y_m = (velocity_y_m - prev_velocity_y_m) / delta_time
                    
                    # MÉTODO MEJORADO: Calcular velocidades y aceleraciones usando espaciado
                    if len(data) >= 7:
                        # Crear DataFrame temporal para análisis con espaciado
                        df_temp = pd.DataFrame(data[-7:])  # Usar últimos 7 puntos
                        
                        # Aplicar suavizado Savitzky-Golay a las posiciones
                        if len(df_temp) >= 7:
                            x_smooth = smooth_data(df_temp['x_m'].values)
                            y_smooth = smooth_data(df_temp['y_m'].values)
                            
                            # Calcular velocidades usando diferencias con espaciado
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
                    
                    # Detectar velocidad inicial para predicción
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
                
                # Dibujar el rectángulo del objeto rastreado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Dibujar círculo preciso superpuesto si está habilitado
                if checkboxes["circle_overlay"]["state"]:
                    # Calcular centro actual del círculo basado en el tracker
                    current_center_x = x + w // 2
                    current_center_y = y + h // 2
                    
                    # Dibujar círculo preciso
                    cv2.circle(frame, (current_center_x, current_center_y), precise_radius, (255, 255, 0), 2)
                    cv2.circle(frame, (current_center_x, current_center_y), 3, (255, 255, 0), -1)
                    
                    # Mostrar información del círculo preciso
                    cv2.putText(frame, f"Radio preciso: {precise_radius}px", 
                               (current_center_x + precise_radius + 10, current_center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Dibujar vectores y resto de visualizaciones (código similar al original)
                center_x = adjusted_x_px
                center_y = frame_height - adjusted_y_px
                
                # [El resto del código de dibujado de vectores permanece igual...]
                # Vectores básicos (deshabilitados por defecto)
                if checkboxes["velocity"]["state"] and prev_x_m is not None:
                    velocity_scale = 50
                    velocity_end_x = int(center_x + velocity_x_m * velocity_scale)
                    velocity_end_y = int(center_y - velocity_y_m * velocity_scale)
                    cv2.arrowedLine(frame, (center_x, center_y), (velocity_end_x, velocity_end_y), (255, 0, 0), 2, tipLength=0.3)
                
                if checkboxes["acceleration"]["state"] and prev_x_m is not None:
                    acceleration_scale = 10
                    accel_end_x = int(center_x + acceleration_x_m * acceleration_scale)
                    accel_end_y = int(center_y - acceleration_y_m * acceleration_scale)
                    cv2.arrowedLine(frame, (center_x, center_y), (accel_end_x, accel_end_y), (0, 0, 255), 2, tipLength=0.3)
                
                if checkboxes["smooth_vectors"]["state"] and len(data) > 7:
                    velocity_scale_smooth = 50
                    acceleration_scale_smooth = 10
                    
                    if checkboxes["y_components"]["state"]:
                        # Solo componente Y
                        velocity_smooth_end_x = center_x
                        velocity_smooth_end_y = int(center_y - smooth_velocity_y * velocity_scale_smooth)
                        cv2.arrowedLine(frame, (center_x, center_y), (velocity_smooth_end_x, velocity_smooth_end_y), (255, 255, 0), 3, tipLength=0.3)
                        
                        accel_smooth_end_x = center_x
                        accel_smooth_end_y = int(center_y - smooth_acceleration_y * acceleration_scale_smooth)
                        cv2.arrowedLine(frame, (center_x, center_y), (accel_smooth_end_x, accel_smooth_end_y), (255, 0, 255), 3, tipLength=0.3)
                    else:
                        # Vector completo
                        velocity_smooth_end_x = int(center_x + smooth_velocity_x * velocity_scale_smooth)
                        velocity_smooth_end_y = int(center_y - smooth_velocity_y * velocity_scale_smooth)
                        cv2.arrowedLine(frame, (center_x, center_y), (velocity_smooth_end_x, velocity_smooth_end_y), (255, 255, 0), 3, tipLength=0.3)
                        
                        accel_smooth_end_x = int(center_x + smooth_acceleration_x * acceleration_scale_smooth)
                        accel_smooth_end_y = int(center_y - smooth_acceleration_y * acceleration_scale_smooth)
                        cv2.arrowedLine(frame, (center_x, center_y), (accel_smooth_end_x, accel_smooth_end_y), (255, 0, 255), 3, tipLength=0.3)
                
                # Predicción de trayectoria
                if checkboxes["prediction"]["state"] and initial_velocity_detected and len(predicted_trajectory_x) > 0:
                    for i in range(len(predicted_trajectory_x) - 1):
                        x1_pred = int((predicted_trajectory_x[i] / pixels_to_meters))
                        y1_pred = int(frame_height - ((predicted_trajectory_y[i] + offset_y_m) / pixels_to_meters))
                        x2_pred = int((predicted_trajectory_x[i+1] / pixels_to_meters))
                        y2_pred = int(frame_height - ((predicted_trajectory_y[i+1] + offset_y_m) / pixels_to_meters))
                        
                        if (0 <= x1_pred < frame_width and 0 <= y1_pred < frame_height and
                            0 <= x2_pred < frame_width and 0 <= y2_pred < frame_height):
                            cv2.line(frame, (x1_pred, y1_pred), (x2_pred, y2_pred), (0, 165, 255), 2)
                
                # Mostrar magnitudes
                if checkboxes["magnitudes"]["state"]:
                    cv2.putText(frame, f"X: {adjusted_x_m:.2f} m", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Y: {adjusted_y_m:.2f} m", (frame_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Mostrar precisión de la medición
                    cv2.putText(frame, f"Precision: {pixels_to_meters*1000:.2f} mm/px", (frame_width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    if prev_x_m is not None:
                        if len(data) > 7:
                            if checkboxes["y_components"]["state"]:
                                cv2.putText(frame, f"Vy_suave: {smooth_velocity_y:.2f} m/s", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                                cv2.putText(frame, f"Ay_suave: {smooth_acceleration_y:.2f} m/s^2", (frame_width - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            else:
                                cv2.putText(frame, f"Vx_suave: {smooth_velocity_x:.2f} m/s", (frame_width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                cv2.putText(frame, f"Vy_suave: {smooth_velocity_y:.2f} m/s", (frame_width - 200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                cv2.putText(frame, f"Ax_suave: {smooth_acceleration_x:.2f} m/s^2", (frame_width - 200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                                cv2.putText(frame, f"Ay_suave: {smooth_acceleration_y:.2f} m/s^2", (frame_width - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                        if initial_velocity_detected:
                            cv2.putText(frame, f"V0 detectada!", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            if checkboxes["y_components"]["state"]:
                                cv2.putText(frame, f"V0y: {initial_vy:.2f} m/s", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, f"V0x: {initial_vx:.2f}, V0y: {initial_vy:.2f}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Actualizar variables previas
                prev_x_m = adjusted_x_m
                prev_y_m = adjusted_y_m
                prev_time = current_time
                if prev_x_m is not None:
                    prev_velocity_x_m = velocity_x_m
                    prev_velocity_y_m = velocity_y_m
                
                # Guardar datos
                data.append({
                    "nro_frame": frame_count,
                    "x_m": round(adjusted_x_m, 4),
                    "y_m": round(adjusted_y_m, 4),
                    "vx_m/s": round(velocity_x_m, 4),
                    "vy_m/s": round(velocity_y_m, 4),
                    "ax_m/s^2": round(acceleration_x_m, 4),
                    "ay_m/s^2": round(acceleration_y_m, 4)
                })
            
            frame_count += 1
        
        # Dibujar checkboxes
        for key, checkbox in checkboxes.items():
            cx, cy = checkbox["pos"]
            cw, ch = checkbox["size"]
            color = (0, 255, 0) if checkbox["state"] else (0, 0, 255)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), color, -1)
            cv2.putText(frame, checkbox["label"], (cx + cw + 5, cy + ch - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Rastreo CSRT - Selección Precisa", frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Crear DataFrame y análisis
    df = pd.DataFrame(data)
    
    print("\n" + "="*50)
    print("ANÁLISIS CON SELECCIÓN PRECISA")
    print("="*50)
    
    # Aplicar análisis mejorado
    df_improved = calculate_derivatives_with_spacing(df, spacing=3)
    
    # Encontrar punto crítico
    critical_frame, critical_idx = find_critical_time(df_improved)
    if critical_frame is not None:
        print(f"Punto crítico encontrado en frame: {critical_frame}")
        print(f"Velocidad Y máxima: {df_improved.loc[critical_idx, 'vy_smooth']:.3f} m/s")
    
    # Análisis de caída libre
    gravity_params, gravity_estimate = analyze_free_fall(df_improved, critical_frame)
    if gravity_estimate is not None:
        print(f"Gravedad estimada: {gravity_estimate:.3f} m/s²")
        print(f"Error respecto a 9.81 m/s²: {abs(gravity_estimate - 9.81):.3f} m/s²")
        print(f"Error porcentual: {abs(gravity_estimate - 9.81)/9.81*100:.1f}%")
    
    # Guardar resultados precisos
    output_file = "resultados_precisos.csv"
    df_improved.to_csv(output_file, index=False)
    print(f"\nDatos precisos guardados en: {output_file}")
    
    print(f"Precisión de medición: {pixels_to_meters*1000:.3f} mm/píxel")
    print(f"Radio usado: {precise_radius} píxeles")
    print(f"Diámetro usado: {pixels_diameter} píxeles")
    print("="*50)

if __name__ == "__main__":
    main()
