# 📊 Análisis de Comentarios con NLP

## 📌 Descripción

Este proyecto implementa un sistema de **Procesamiento de Lenguaje Natural (NLP)** para analizar comentarios de usuarios sobre productos (en este caso, un portátil Lenovo).

El sistema realiza:
- Preprocesamiento de texto
- Análisis lingüístico
- Extracción de entidades (NER)
- Análisis de sentimiento
- Descubrimiento de tópicos

---

## ⚙️ Tecnologías utilizadas

- **SpaCy** (procesamiento principal)
- **NLTK** (VADER para sentimiento)
- **Stanza** (análisis sintáctico y sentimiento)
- **Hugging Face Transformers** (modelo avanzado de sentimiento)
- **TextBlob** (spacytextblob)
- **Polars** (dataframes)
- **Deep Translator** (traducción automática)

---

## 🔄 Funcionamiento general

El script realiza el siguiente flujo:

1. Carga de comentarios desde un archivo `.txt`
2. Preprocesamiento del texto
3. Análisis lingüístico
4. Extracción de entidades
5. Análisis de sentimiento
6. Descubrimiento de tópicos

---

## 🧹 Preprocesamiento

**Función:** `cleaning_words`

- Elimina stopwords  
- Elimina signos de puntuación  
- Obtiene los lemas de las palabras  

**Devuelve:**  
Lista de palabras normalizadas por comentario

---

## 🔍 Análisis lingüístico

Se realiza con **SpaCy** y **Stanza**

### SpaCy
- Tokenización  
- Lematización  
- POS tagging  
- Árbol de dependencias (se guarda en HTML)

### Stanza
- Árbol de constituyentes  
- Análisis de sentimiento por frases  

---

## 🏷️ Extracción de entidades (NER)

**Función:** `extract_entities`

- Detecta entidades nombradas  
- Filtra entidades irrelevantes  
- Añade descripciones automáticas y personalizadas  

Incluye soporte para siglas mediante `EntityRuler`.

**Devuelve:**  
DataFrame con:
- Entidad  
- Etiqueta  
- Descripción  

---

## 😊 Análisis de sentimiento

**Función:** `sentiment_analysis`

Se utilizan 4 modelos:

### 1. TextBlob (SpaCy)
- > 0 → positivo  
- = 0 → neutral  
- < 0 → negativo  

### 2. Stanza
- < 1 → negativo  
- ≈ 1 → neutral  
- > 1 → positivo  

### 3. VADER (NLTK)
- ≥ 0.5 → positivo  
- entre -0.05 y 0.05 → neutral  
- < -0.05 → negativo  

### 4. Hugging Face
- Devuelve etiqueta (positivo, negativo, neutral)  
- Devuelve score de confianza  

**Resultados:**  
DataFrame con todas las métricas

**Conclusión del análisis:**
- Mayoría de comentarios positivos  
- Hugging Face es el modelo más preciso en contexto  

---

## 🧠 Descubrimiento de tópicos

### 🔹 Estrategia A: Frecuencia

**Funciones:**
- `extract_noun_chunks`
- `extract_nouns`

**Resultados:**
Palabras más repetidas:
- precio  
- características  
- ordenador  
- portátil  

---

### 🔹 Estrategia B: Similitud semántica

**Función:** `get_similarities`

Se compara cada comentario con palabras clave:
- política  
- deportes  
- tecnología  
- cine  
- portátil  

**Resultado:**
La categoría más frecuente es **tecnología / portátil**

---

## 🧪 Ejecución

### Instalar dependencias

```bash
pip install spacy nltk stanza transformers polars deep-translator spacytextblob
python -m spacy download es_core_news_lg
```

### Ejecutar

```bash
python AnalysisSentimentLaptop.py
```

## 📄 Entrada

Archivo `.txt` con un comentario por línea.

**Ejemplo:**
- Satisfecho con la compra.
- Excelente precio.
- Muy buen portátil.


---

## 📤 Salida

### Consola
- Tokens procesados  
- Etiquetas gramaticales  
- Entidades  
- Sentimientos  
- Tópicos  

### Archivo generado
- `arbolDependencia-AGG.html`

---

## 📌 Conclusiones

- La mayoría de comentarios son positivos  

- Se valoran especialmente:
  - Precio  
  - Rendimiento  
  - Características técnicas  

- Principales quejas:
  - Estado del producto al recibirlo  

- Tema principal:
  - Tecnología (portátiles)  

---

## 👤 Autor

Adrián García García
