# Proyecto de Machine Learning

## 📌 Descripción
Este proyecto implementa un flujo de trabajo básico de Machine Learning en Python. Incluye la carga de datos, la construcción de un modelo, el entrenamiento y la evaluación del mismo.

## 📂 Estructura del Proyecto
```
📂 proyecto_ml/
├── 📂 src/
│   ├── data.py          # Funciones para la carga de datos
│   ├── model.py         # Definición del modelo
│   ├── train.py         # Función para entrenar el modelo
│   ├── evaluate.py      # Función para evaluar el modelo
├── main.py              # Script principal
├── README.md            # Documentación del proyecto
```

## 📦 Instalación
Asegúrate de tener Python instalado y ejecuta el siguiente comando para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## 🚀 Uso
Ejecuta el siguiente comando en la terminal para iniciar el proceso de entrenamiento y evaluación:

```bash
python main.py
```

## 🛠️ Funcionalidad
1. **Carga de datos**: Se importan los datos utilizando `load_data()` desde `data.py`.
2. **Construcción del modelo**: Se define un modelo de Machine Learning con `build_model()` desde `model.py`.
3. **Entrenamiento**: Se entrena el modelo usando `train_model()` desde `train.py`.
4. **Evaluación**: Se evalúa el modelo con `evaluate_model()` desde `evaluate.py`.



