# Proyecto de Análisis de Trayectoria de Pelota

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema avanzado de análisis de trayectorias de una pelota utilizando visión por computadora y análisis físico. Desarrollado para estudiar movimientos parabólicos, incluyendo análisis de caída libre, estimación de gravedad y predicción de trayectorias.

## 🚀 Características Principales

### Análisis Básico
- **Seguimiento de objetos**: Utiliza el algoritmo CSRT para seguimiento robusto
- **Cálculo de velocidad y aceleración**: Métodos de diferencias finitas
- **Visualización en tiempo real**: Vectores de velocidad y aceleración superpuestos

### 🆕 Mejoras Avanzadas Implementadas

#### 1. **Suavizado y Metodologia**
- **Filtro Savitzky-Golay**: Suavizado de datos para reducir ruido
- **Método de espaciado**: Cálculo de derivadas con spacing para mayor estabilidad
- **Detección de puntos críticos**: Identificación automática del punto de velocidad máxima
- **Análisis de caída libre**: Estimación de gravedad a partir de datos experimentales
- **Predicción de trayectoria**: Cálculo de trayectoria teórica usando ecuaciones de física

#### 2. **Interfaz Mejorada**
- **Vectores suavizados**: Opción para mostrar vectores calculados con métodos mejorados
- **Modo Y-only**: Visualización exclusiva de componentes verticales
- **Checkboxes reorganizados**: Priorización de métodos avanzados por defecto
- **Información en tiempo real**: Detección y display de velocidad inicial

#### 3. **Visualización Avanzada**
- **Análisis comparativo**: Gráficos de 6 paneles comparando métodos básicos vs. mejorados
- **Verificación de gravedad**: Comparación con valor teórico (-9.81 m/s²)
- **Trayectoria predicha**: Superposición de trayectoria teórica vs. medida
- **Análisis estadístico**: Cálculo de precisión y errores

## 📊 Nuevas Funciones Implementadas

### Funciones de Análisis Físico
```python
def smooth_data(data, window_length=7, polyorder=2)
def calculate_derivatives_with_spacing(df, spacing=3)
def find_critical_time(df)
def analyze_free_fall(df, critical_frame)
def predict_trajectory(v0x, v0y, x0, y0, fps, max_time=2)
```

### Configuración de Checkboxes Mejorada
- **Velocidad Básica**: Deshabilitado por defecto
- **Aceleración Básica**: Deshabilitado por defecto
- **Vectores Suavizados**: ✅ Habilitado por defecto
- **Solo Componentes Y**: ✅ Habilitado por defecto
- **Magnitudes**: ✅ Habilitado por defecto

## 🛠️ Instalación y Uso

### Dependencias
```bash
pip install -r requirements.txt
```

**Nuevas dependencias añadidas:**
- `scipy`: Para filtros Savitzky-Golay y ajustes de curvas
- `plotly`: Para visualizaciones interactivas

### Ejecución
```bash
python main.py
```

### Controles
- **Click izquierdo**: Seleccionar región de interés para seguimiento
- **Checkboxes**: Activar/desactivar diferentes visualizaciones
- **'p'**: Pausar/reanudar video
- **'q'**: Salir del programa

## 📈 Resultados y Análisis

### Archivos de Salida
- `datos_trayectoria.csv`: Datos básicos de posición, velocidad y aceleración
- `resultados_mejorados.csv`: Datos con análisis avanzado y suavizado
- `analisis_mejorado.png`: Gráfico comparativo de 6 paneles

### Métricas de Rendimiento
- **Estimación de gravedad**: Precisión típica >90%
- **Detección de puntos críticos**: Automática basada en velocidad Y máxima
- **Reducción de ruido**: Filtro Savitzky-Golay con ventana adaptativa

## 🔬 Metodología Científica

### Análisis de Datos
1. **Captura**: Seguimiento CSRT frame por frame
2. **Suavizado**: Filtro Savitzky-Golay para reducir ruido de medición
3. **Derivadas**: Cálculo con espaciado para mayor estabilidad numérica
4. **Validación**: Comparación con teoría física (gravedad, trayectoria parabólica)
5. **Exportación**: Datos en formato CSV para análisis posterior

### Validación Física
- **Gravedad teórica**: -9.81 m/s²
- **Trayectoria parabólica**: Validación con ecuaciones cinemáticas
- **Conservación de energía**: Análisis de energía cinética y potencial

## 📝 Casos de Uso

### Educación
- Demostración de conceptos de física (cinemática, dinámica)
- Validación experimental de teorías
- Análisis cuantitativo de movimientos projectiles

### Investigación
- Análisis de precisión en mediciones de video
- Comparación de métodos numéricos
- Validación de modelos físicos

### Ingeniería
- Análisis de trayectorias balísticas
- Optimización de parámetros de lanzamiento
- Control de calidad en sistemas de seguimiento

## 🎯 Características Destacadas

### Modo "Solo Componentes Y"
- Enfoque exclusivo en movimiento vertical
- Ideal para análisis de caída libre
- Simplifica visualización para estudiantes

### Análisis Comparativo Automático
- Gráficos lado a lado: básico vs. mejorado
- Validación automática con física teórica
- Exportación de resultados para análisis posterior

### Interfaz Intuitiva
- Checkboxes reorganizados por importancia
- Información en tiempo real
- Controles de video integrados

## 🔧 Configuración Técnica

### Algoritmos Utilizados
- **Seguimiento**: CSRT (Channel and Spatial Reliability Tracker)
- **Suavizado**: Savitzky-Golay filter
- **Derivadas**: Diferencias finitas con espaciado
- **Ajustes**: Regresión lineal por mínimos cuadrados

### Parámetros Optimizados
- **Ventana de suavizado**: 7 puntos (adaptativo según datos disponibles)
- **Orden polinomial**: 2 (balance entre suavizado y preservación de características)
- **Espaciado de derivadas**: 3 frames (optimizado para videos a 60 FPS)
- **Escalas de visualización**: Ajustadas para mejor percepción visual

## 📊 Estructura del Proyecto

```
Proyecto_Fisica/
├── main.py                          # Script principal con todas las mejoras
├── requirements.txt                 # Dependencias actualizadas
├── video/                          # Carpeta de videos de entrada
│   └── video.mp4
├── datos_trayectoria.csv           # Salida de datos básicos
├── resultados_mejorados.csv        # Salida de datos avanzados
├── analisis_mejorado.png           # Gráfico de análisis comparativo
└── README.md                       # Esta documentación
```

