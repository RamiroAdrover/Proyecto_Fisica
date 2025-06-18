import cv2
import numpy as np
import json

class CircleSelector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.original_frame = None
        self.display_frame = None
        self.center = None
        self.radius = None
        self.drawing = False
        self.selection_complete = False
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_pos = None
        
        # Configuración de la interfaz
        self.window_name = "Selector Preciso de Círculo"
        self.info_height = 150
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.selection_complete:
                # Convertir coordenadas de pantalla a coordenadas de imagen
                img_x, img_y = self.screen_to_image_coords(x, y)
                
                if self.center is None:
                    # Primer clic: establecer centro
                    self.center = (img_x, img_y)
                    self.drawing = True
                else:
                    # Segundo clic: establecer radio y completar selección
                    self.radius = int(np.sqrt((img_x - self.center[0])**2 + (img_y - self.center[1])**2))
                    self.drawing = False
                    self.selection_complete = True
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.center is not None:
                # Actualizar radio temporal mientras se arrastra
                img_x, img_y = self.screen_to_image_coords(x, y)
                self.radius = int(np.sqrt((img_x - self.center[0])**2 + (img_y - self.center[1])**2))
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic derecho: reiniciar selección
            self.reset_selection()
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Clic medio: iniciar pan
            self.last_mouse_pos = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_MBUTTON:
            # Arrastrar con botón medio: pan
            if self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.pan_x += dx / self.zoom_factor
                self.pan_y += dy / self.zoom_factor
                self.last_mouse_pos = (x, y)
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Rueda del mouse: zoom
            zoom_delta = 0.1 if flags > 0 else -0.1
            new_zoom = max(0.5, min(5.0, self.zoom_factor + zoom_delta))
            
            # Ajustar pan para zoom centrado en el cursor
            img_x, img_y = self.screen_to_image_coords(x, y)
            self.zoom_factor = new_zoom
            new_img_x, new_img_y = self.screen_to_image_coords(x, y)
            self.pan_x += new_img_x - img_x
            self.pan_y += new_img_y - img_y
    
    def screen_to_image_coords(self, screen_x, screen_y):
        """Convierte coordenadas de pantalla a coordenadas de imagen"""
        # Ajustar por la altura del panel de información
        screen_y -= self.info_height
        
        # Convertir a coordenadas de imagen
        img_x = int((screen_x / self.zoom_factor) - self.pan_x)
        img_y = int((screen_y / self.zoom_factor) - self.pan_y)
        
        return img_x, img_y
    
    def image_to_screen_coords(self, img_x, img_y):
        """Convierte coordenadas de imagen a coordenadas de pantalla"""
        screen_x = int((img_x + self.pan_x) * self.zoom_factor)
        screen_y = int((img_y + self.pan_y) * self.zoom_factor) + self.info_height
        
        return screen_x, screen_y
    
    def reset_selection(self):
        """Reinicia la selección"""
        self.center = None
        self.radius = None
        self.drawing = False
        self.selection_complete = False
    
    def update_display(self):
        """Actualiza la imagen mostrada"""
        if self.original_frame is None:
            return
            
        # Crear frame de trabajo
        frame = self.original_frame.copy()
        
        # Aplicar zoom y pan
        h, w = frame.shape[:2]
        
        # Calcular región visible
        x1 = max(0, int(-self.pan_x))
        y1 = max(0, int(-self.pan_y))
        x2 = min(w, int(w/self.zoom_factor - self.pan_x))
        y2 = min(h, int(h/self.zoom_factor - self.pan_y))
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            frame = cv2.resize(roi, (int((x2-x1)*self.zoom_factor), int((y2-y1)*self.zoom_factor)))
        
        # Dibujar círculo si está definido
        if self.center is not None and self.radius is not None:
            # Convertir coordenadas del círculo a coordenadas de pantalla
            center_screen = self.image_to_screen_coords(self.center[0], self.center[1])
            center_screen = (center_screen[0], center_screen[1] - self.info_height)
            radius_screen = int(self.radius * self.zoom_factor)
            
            # Verificar si el círculo está visible
            h_display, w_display = frame.shape[:2]
            if (0 <= center_screen[0] <= w_display and 
                0 <= center_screen[1] <= h_display):
                
                # Dibujar círculo principal
                color = (0, 255, 0) if self.selection_complete else (0, 255, 255)
                cv2.circle(frame, center_screen, radius_screen, color, 2)
                
                # Dibujar centro
                cv2.circle(frame, center_screen, 3, (0, 0, 255), -1)
                
                # Dibujar líneas de referencia
                cv2.line(frame, (center_screen[0]-10, center_screen[1]), 
                        (center_screen[0]+10, center_screen[1]), (0, 0, 255), 1)
                cv2.line(frame, (center_screen[0], center_screen[1]-10), 
                        (center_screen[0], center_screen[1]+10), (0, 0, 255), 1)
                
                # Dibujar radio
                if radius_screen > 0:
                    end_point = (center_screen[0] + radius_screen, center_screen[1])
                    cv2.line(frame, center_screen, end_point, (255, 0, 0), 1)
                    
                    # Mostrar medida del radio
                    cv2.putText(frame, f"{self.radius}px", 
                               (center_screen[0] + radius_screen//2, center_screen[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Crear panel de información
        info_panel = np.zeros((self.info_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Información de estado
        if self.center is None:
            status = "1. Haz clic en el CENTRO del círculo"
        elif not self.selection_complete:
            status = "2. Haz clic para definir el RADIO del círculo"
        else:
            status = "¡SELECCIÓN COMPLETA! Presiona ENTER para confirmar"
        
        cv2.putText(info_panel, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controles
        controls = [
            "CONTROLES:",
            "- Rueda del mouse: Zoom",
            "- Boton medio + arrastrar: Pan",
            "- Clic derecho: Reiniciar seleccion",
            "- R: Reiniciar seleccion",
            "- ENTER: Confirmar seleccion",
            "- ESC: Salir"
        ]
        
        for i, control in enumerate(controls):
            y_pos = 45 + i * 15
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(info_panel, control, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Información del círculo
        if self.center is not None:
            circle_info = f"Centro: ({self.center[0]}, {self.center[1]})"
            cv2.putText(info_panel, circle_info, (400, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if self.radius is not None:
                radius_info = f"Radio: {self.radius} pixeles"
                cv2.putText(info_panel, radius_info, (400, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Calcular bbox
                bbox = (self.center[0] - self.radius, self.center[1] - self.radius, 
                       2 * self.radius, 2 * self.radius)
                bbox_info = f"BBox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                cv2.putText(info_panel, bbox_info, (400, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Información de zoom
        zoom_info = f"Zoom: {self.zoom_factor:.1f}x"
        cv2.putText(info_panel, zoom_info, (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combinar panel de información con imagen
        self.display_frame = np.vstack([info_panel, frame])
    
    def run(self):
        """Ejecuta el selector"""
        if not self.cap.isOpened():
            print("Error: No se pudo abrir el video")
            return None
            
        # Leer primer frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: No se pudo leer el primer frame")
            return None
            
        self.original_frame = frame.copy()
        
        # Configurar ventana
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("=== SELECTOR PRECISO DE CÍRCULO ===")
        print("INSTRUCCIONES:")
        print("1. Haz clic en el CENTRO del círculo que quieres rastrear")
        print("2. Haz clic en el BORDE para definir el radio")
        print("3. Usa la rueda del mouse para hacer zoom")
        print("4. Mantén presionado el botón medio y arrastra para hacer pan")
        print("5. Clic derecho para reiniciar la selección")
        print("6. Presiona ENTER para confirmar o ESC para salir")
        print("="*50)
        
        while True:
            self.update_display()
            
            if self.display_frame is not None:
                cv2.imshow(self.window_name, self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Selección cancelada")
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_selection()
                print("Selección reiniciada")
            elif key == 13:  # ENTER
                if self.selection_complete:
                    result = self.get_selection_data()
                    print("¡Selección confirmada!")
                    print(f"Centro: {result['center']}")
                    print(f"Radio: {result['radius']}")
                    print(f"BBox: {result['bbox']}")
                    cv2.destroyAllWindows()
                    return result
                else:
                    print("Completa la selección primero (centro y radio)")
        
        cv2.destroyAllWindows()
        return None
    
    def get_selection_data(self):
        """Obtiene los datos de la selección"""
        if not self.selection_complete:
            return None
            
        bbox = (self.center[0] - self.radius, self.center[1] - self.radius, 
               2 * self.radius, 2 * self.radius)
        
        return {
            'center': self.center,
            'radius': self.radius,
            'bbox': bbox,
            'pixels_diameter': 2 * self.radius
        }
    
    def __del__(self):
        if self.cap is not None:
            self.cap.release()

def main():
    # Configuración
    video_path = "video/video.mp4"
    output_file = "seleccion_precisa.json"
    
    # Crear selector
    selector = CircleSelector(video_path)
    
    # Ejecutar selección
    result = selector.run()
    
    if result is not None:
        # Guardar resultado
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nDatos guardados en: {output_file}")
        print("\nPuedes usar estos datos en tu programa principal:")
        print(f"Centro: {result['center']}")
        print(f"Radio: {result['radius']}")
        print(f"BBox para tracker: {result['bbox']}")
        print(f"Diámetro en píxeles: {result['pixels_diameter']}")
        
        # Calcular relación píxeles a metros
        ball_diameter_m = 0.24  # Diámetro de la pelota en metros
        pixels_to_meters = ball_diameter_m / result['pixels_diameter']
        print(f"Relación píxeles a metros: {pixels_to_meters:.6f} m/px")
        
        return result
    else:
        print("No se realizó ninguna selección")
        return None

if __name__ == "__main__":
    main()
