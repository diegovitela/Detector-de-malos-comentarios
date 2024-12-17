from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import re

# Descargar stopwords en español
nltk.download('stopwords')

# Crear la aplicación Flask
app = Flask(__name__)

# Dataset de entrenamiento
data = {
    'texto': [
        # Recomendaciones claras
        "Me encantó el producto, muy recomendado",
        "El servicio al cliente fue excelente",
        "Muy satisfecho con mi compra",
        "Fantástica experiencia de compra",
        "Atención rápida y eficiente",
        "Buen producto, entrega rápida",
        "Increíble servicio, lo recomiendo",
        "Excelente calidad y atención",
        "Calidad perfecta, lo volveré a comprar",
        "Todo fue excelente, muy recomendable",
        "Gran experiencia, muy satisfecho",
        "Servicio y calidad impecables",
        "Producto excelente, súper recomendado",
        "La mejor compra que he hecho",
        "Buen precio y calidad, todo perfecto",

        # Quejas claras
        "No estoy contento con el producto",
        "El envío fue muy lento",
        "La calidad es terrible, no lo recomiendo",
        "Producto defectuoso, necesito un reembolso",
        "No volveré a comprar aquí, pésima atención",
        "El producto llegó roto, pésimo servicio",
        "El peor servicio que he recibido",
        "Nunca había tenido una experiencia tan mala",
        "Es horrible, no me gustó nada",
        "Este servicio es una basura, jamás lo recomendaría",
        "Producto malo, no lo volveré a comprar",
        "La experiencia fue pésima",
        "La atención al cliente fue pésima",
        "Producto muy caro para lo que ofrece",
        "Mala experiencia, no lo recomiendo",

        # Casos ambiguos
        "El producto es bueno, pero el servicio no",
        "Me gustó el servicio, pero podría mejorar",
        "Buena calidad, aunque llegó tarde",
        "El producto está bien, pero no es lo que esperaba",
        "La calidad es aceptable, pero el precio es alto",
        "Servicio aceptable, aunque tardaron en responder",
        "La atención fue amable, pero el producto no funcionó",
        "Me gustó el precio, pero el envío fue lento",
        "El producto cumple, pero esperaba más",
        "El servicio no fue malo, pero tampoco destacable",

        # Recomendaciones adicionales
        "Excelente trato, calidad y precio",
        "Muy buena calidad, seguiré comprando",
        "Todo llegó a tiempo y en perfectas condiciones",
        "Mejor de lo que esperaba, excelente experiencia",
        "Totalmente satisfecho con mi compra",
        "Rápido, eficiente y confiable, muy recomendado",
        "Gran calidad y atención al cliente",
        "Perfecto para lo que necesitaba, excelente servicio",
        "Increíble atención y producto de calidad",
        "Superó todas mis expectativas, lo recomiendo ampliamente",

        # Quejas adicionales
        "El producto llegó dañado, muy mala experiencia",
        "La atención fue grosera y poco profesional",
        "Muy mala calidad, no cumple lo que promete",
        "El servicio fue lento y poco eficiente",
        "Tuve que pedir un reembolso, no satisfecho",
        "El peor producto que he comprado, decepcionante",
        "Nada salió bien, fue una mala compra",
        "La experiencia fue frustrante, no lo recomiendo",
        "El servicio fue una pérdida de tiempo",
        "No cumple con lo descrito, muy mal producto"
    ],
    'categoria': [
        # Etiquetas correspondientes (ordenadas)
        'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion',
        'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion',
        'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion',
        'queja', 'queja', 'queja', 'queja', 'queja',
        'queja', 'queja', 'queja', 'queja', 'queja',
        'queja', 'queja', 'queja', 'queja', 'queja',
        'ambigua', 'ambigua', 'ambigua', 'ambigua', 'ambigua',
        'ambigua', 'ambigua', 'ambigua', 'ambigua', 'ambigua',
        'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion',
        'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion', 'recomendacion',
        'queja', 'queja', 'queja', 'queja', 'queja',
        'queja', 'queja', 'queja', 'queja', 'queja'
    ]
}

df = pd.DataFrame(data)
X = df['texto']
y = df['categoria']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Preprocesamiento: stopwords en español
stop_words_es = stopwords.words('spanish')

# Modelo de pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words=stop_words_es)),
    ('nb', MultinomialNB(alpha=0.5))
])

# Entrenar el modelo
model.fit(X_train, y_train)

def limpiar_texto(texto):
    """
    Limpia el texto eliminando puntuación y transformando a minúsculas.
    """
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

# Clasificar texto
def clasificar_texto(texto):
    """
    Clasifica un texto dado y retorna la categoría y probabilidades.
    """
    texto_limpio = limpiar_texto(texto)
    probabilidad = model.predict_proba([texto_limpio])[0]
    categorias = model.classes_
    prediccion = categorias[probabilidad.argmax()]
    return prediccion, dict(zip(categorias, probabilidad))

@app.route("/", methods=["GET", "POST"])
def index():
    categoria = None
    probabilidades = None
    texto_usuario = None

    if request.method == "POST":
        texto_usuario = request.form["texto"]
        categoria, probabilidades = clasificar_texto(texto_usuario)

    return render_template("index.html", categoria=categoria, probabilidades=probabilidades, texto=texto_usuario)

if __name__ == "__main__":
    app.run(debug=True)

