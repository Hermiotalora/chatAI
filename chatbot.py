from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
from transformers import pipeline
import torch
import torch.nn as nn

# Cargar variables de entorno
load_dotenv()

# Configurar claves de API y modelos externos
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar Flask app
app = Flask(__name__)

# Inicializar modelos de Hugging Face y PyTorch
huggingface_classifier = pipeline("sentiment-analysis")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

pytorch_model = SimpleNN()

@app.route('/chat', methods=['POST'])
def chat_with_openai():
    data = request.json
    user_input = data.get("message", "")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": user_input}
            ]
        )
        return jsonify({"response": response.choices[0].message['content']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    data = request.json
    text = data.get("text", "")
    try:
        result = huggingface_classifier(text)
        return jsonify({"sentiment": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pytorch', methods=['POST'])
def run_pytorch_model():
    try:
        # Generar datos de entrada aleatorios para el modelo
        sample_input = torch.rand(1, 10)
        output = pytorch_model(sample_input)
        return jsonify({"output": output.item()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


'''#Primer Ejemplo Simple

import openai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la clave de API desde las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

print("Bienvenido al Chatbot IA. Escribe 'salir' para terminar.")

while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        print("¡Hasta luego!")
        break

    try:
        # Usar la API para obtener la respuesta
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambia el modelo según lo que uses
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": user_input}
            ]
        )
        # Imprimir la respuesta del chatbot
        print(f"Chatbot: {response.choices[0].message['content']}")
    except Exception as e:
        print(f"Error al comunicarse con la API:\n\n{e}")

'''
'''
#Segundo Ejemplo simple

import openai
import os
from dotenv import load_dotenv
from transformers import pipeline
import torch
import torch.nn as nn

# Cargar variables de entorno
load_dotenv()

# Obtener la clave de API desde las variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cargar un modelo de Hugging Face para clasificación de texto
classifier = pipeline("sentiment-analysis")  # Puedes cambiar el modelo según la tarea

# Definir un modelo simple en PyTorch
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 1)  # Ejemplo: una capa lineal

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Crear una instancia del modelo
pytorch_model = SimpleNN()

print("Bienvenido al Chatbot IA. Escribe 'salir' para terminar.")
print("Opciones adicionales:")
print("1. Análisis de sentimiento con Hugging Face")
print("2. Ejecucion de un modelo PyTorch")

while True:
    user_input = input("\nTú: ")
    if user_input.lower() == "salir":
        print("¡Hasta luego!")
        break

    if user_input == "1":
        text = input("Escribe un texto para analizar su sentimiento: ")
        result = classifier(text)
        print(f"Análisis de sentimiento: {result}")
    elif user_input == "2":
        print("Ejecución del modelo PyTorch:")
        sample_input = torch.rand(1, 10)  # Entrada de ejemplo con 10 características
        output = pytorch_model(sample_input)
        print(f"Salida del modelo: {output.item()}")
    else:
        try:
            # Usar la API de OpenAI para obtener la respuesta
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": user_input}
                ]
            )
            # Imprimir la respuesta del chatbot
            print(f"Chatbot: {response.choices[0].message['content']}")
        except Exception as e:
            print(f"Error al comunicarse con la API:\n\n{e}")
'''