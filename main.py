import sys
import os

# Agregar 'src/' al path para poder importar los módulos
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Ahora se pueden importar los módulos sin error
from data import load_data
from model import build_model
from train import train_model
from evaluate import evaluate_model

def main():
    # Cargar datos
    x_train, y_train, x_test, y_test = load_data()

    # Construir modelo
    model = build_model()
    model.summary()

    # Entrenar modelo
    train_model(model, x_train, y_train)
    
    # Evaluar modelo
    evaluate_model(model, x_test, y_test)

    print("Entrenamiento y evaluación completados.")

if __name__ == "__main__":
    main()