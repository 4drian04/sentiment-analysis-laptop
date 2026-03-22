from nltk.data import load
from nltk.corpus import stopwords
import nltk
from nltk import Tree
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from random import Random
import polars as pl
from spacy import displacy
from spacy.tokens import Span
import stanza
from spacytextblob.spacytextblob import SpacyTextBlob
from deep_translator import GoogleTranslator
from spacy.language import Language
from spacy.tokens import Doc
from transformers import pipeline
from spacy.pipeline import EntityRuler

def cleaning_words(doc, nlp):
    """
    Hace un preprocesado de los distintos comentarios pasados como parámetro.
    Además, obtiene los lemas de las distintas palabras del comentario, quitando puntuaciones y stop words.
    En este caso, como usamos spacy, no es necesario añadir al pipeline las funciones como añadir el lema de la palabra,
    ya que en Spacy ya viene incluido (cuando se hace nlp(line)). Donde habría que incluirlo sería en Stanza, ya que ahí creas el pipeline desde cero,
    teniendo que añadir los distintos componentes
    Args:
        doc: Todos los comentarios de un producto.
        nlp: Nlp generado
    Returns:
        list: Devuelve una lista con los lemas de los comentarios.
    """
    cleaningWords = []
    lines = doc.text.splitlines()
    for line in lines:
        tokens = nlp(line)
        normalized_docs = [token.lemma_ for token in tokens if not token.is_punct and not token.is_stop]
        cleaningWords.append(normalized_docs)
    return cleaningWords

# Esto nos permite añadir la librería de Hugging Face al pipeline de nlp
@Language.component("hf_sentiment_component")
def hf_sentiment_component(doc):
    result = classifier(doc.text, truncation=True, max_length=128)[0]
    doc._.hf_sentiment = result["label"]
    doc._.hf_score = result["score"]
    return doc

#------------------------------------------
# Apartado 3. Extracción de Entidades (NER)
#------------------------------------------

def extract_entities(docsReview):
    """
    Se obtiene las distintas entidades de los comentarios (Docs) pasado como parámetro
    Args:
        docReview: Todos los comentarios de un producto.
    Returns:
        ents_token: Devuelve un dataframe con las distintas entidades encontradas en los comentarios
    """
    ents_filtradas = []
    # Iteramos sobre todos los docs
    for doc in docsReview:
        for ent in doc.ents:
            # Filtrado de MISC que no interesa
            if ent.label_ == "MISC" and ent.start == ent.sent.start and ent.text[1:].islower():
                continue

            # Si es SIGLA, asignamos la descripción desde el EntityRuler
            if ent.label_ == "SIGLA":
                pattern_desc = pattern_descriptions.get(ent.ent_id_, "Sigla personalizada")
                ent._.description = pattern_desc
            else:
                ent._.description = spacy.explain(ent.label_)

            ents_filtradas.append(ent)

    # Creamos el DataFrame de las diferentes entidades
    ents_token = pl.DataFrame([{
        "Entidad": ent.text,
        "Etiqueta": ent.label_,
        "Descripción": ent._.description
    } for ent in ents_filtradas])
    return ents_token

#-------------------------------------------------------------
# Apartado 4. Análisis de Sentimiento (El componente "Custom")
#-------------------------------------------------------------

@Language.component("vader_sentiment") # Esto nos permitirá crear un componente para añdirlo al Pipeline
def vader_sentiment(doc):
    scores = sia.polarity_scores(doc.text) # Calcula la polaridad del comentario para ver si es negativo/neutral/positivo
    doc._.vader_scores = scores # Guarda la puntuación
    doc._.vader_compound = scores["compound"] # Guarda la puntuación media
    return doc

def sentiment_analysis(docsReview, nlp, stanza_nlp):
    """
    Se hace un análisis de sentimientos de los distintos comentarios.
    Se hará un análisis con Spacy, Stanza, VADER y Hugging Face, para poder determinar que modelo tiene una mejor precisión
    Args:
        docReview: Todos los comentarios de un producto.
        nlp: Permite aplicarle las distintas funcionalidades al texto traducido
        stanza_nlp: Es el procesamiento natural de stanza
    Returns:
        results: Devuelve una lista de diccionarios con los análisis de los distintos comentarios
    """
    # Vamos a ver si los comentarios son positivos o negativos
    translator = GoogleTranslator(source='auto', target='en') # Creamos el traductor para traducir el comentario del español al inglés
    results = []

    for doc in docsReview:
        # Traducción para modelos en inglés
        try:
            docTranslate = translator.translate(doc.text)
        except Exception: # En caso de que la traducción de alguna excepción, seguimos sin la traducción
            docTranslate=doc.text
        docAux = nlp(docTranslate)
        
        # STANZA
        stanzaAux = stanza_nlp(doc.text)
        sentimientos = [s.sentiment for s in stanzaAux.sentences]
        media_stanza = sum(sentimientos) / len(sentimientos) # Calculamos la media del sentimiento de la reseña
        
        # Guardamos resultados
        results.append({
            "Texto": doc.text,
            "Stanza_media": media_stanza,
            "TextBlob_polarity": docAux._.blob.polarity, # Obtenemos la puntuación de sentimiento del modelo textblob
            "VADER_compound": docAux._.vader_compound, # Obtenemos la puntuación de sentimiento del modelo VADER
            "HF_label": doc._.hf_sentiment, # Obtenemos el sentimiento del modelo Hugging Face creado al principio
            "HF_score": doc._.hf_score
        })
    return results

#--------------------------------------------------------
# Apartado 5. Descubrimiento del Tópico (Topic Discovery)
#--------------------------------------------------------
# En primer lugar vamos a analizar los sintagmas nominales (Noun Chunks)

#doc.noun_chunks: Es el iterador principal que spaCy usa para encontrar frases nominales. Requiere que el texto haya sido analizado sintácticamente (lo que hace spacy.load y la ejecución del doc).
# chunk.text: El texto completo del sintagma nominal ("El gato rápido").
# chunk.root.text: La palabra núcleo del sintagma ("gato").
# chunk.root.dep_: La relación de dependencia del núcleo con respecto al resto de la frase (ej. nsubj para sujeto nominal, dobj para objeto directo).
# chunk.root.head.text: La palabra de la que depende el núcleo (generalmente el verbo principal del sintagma verbal).
# Creamos un set con las stopwords en español

def extract_noun_chunks(docsReview):
    """
    Obtenemos los distintos sintagmas nominales, así como el núcleo del sintagma, la relación de dependencia del núcleo y la palabra que depende del núcleo
    Args:
        docReview: Todos los comentarios de un producto.
    Returns:
        chunksDf: Dataframe que nos permite ver los distintos sintagmas nominales
    """
    stop_words_es = set(stopwords.words('spanish')) # Creamos el conjunto de stopwords en español
    # DataFrame filtrando directamente al crear la lista
    chunksDf = pl.DataFrame([
        {
            "Sintagma": chunk.text,
            "Núcleo": chunk.root.text,
            "Dependencia": chunk.root.dep_,
            "Cabeza": chunk.root.head.text
        }
        for doc in docsReview
        for chunk in doc.noun_chunks
        if not all(token.lower() in stop_words_es for token in chunk.text.split())
    ])
    return chunksDf

def extract_nouns(docsReview):
    """
    Se extrae los distintos sustantivos de los distintos comentarios pasado como parámetros
    Args:
        docReview: Todos los comentarios de un producto.
    Returns:
        pos_nouns: Devuelve una lista con los distintos sustantivos
    """
    # Obtenemos ahora los sustantivos de los comentarios
    pos_nouns = pl.DataFrame([{
        "Texto": noun.text,
        "POS": noun.pos_,
        "Lemma": noun.lemma_
    } for doc in docsReview for noun in doc if noun.pos_ in ['NOUN']])
    return pos_nouns


# Estrategia B

def get_similarities(docsReview, keyWords):
    """
    Obtiene la similitud con las distintas palabras pasadas como parámetro (lista)
    Args:
        docReview: Todos los comentarios de un producto.
        keyWords: Las listas de str para hacer la similitud con un comentario en cuestión
    Returns:
        results: Devuelve una lista de diccionarios con las similitudes de los distintos comentarios
    """
    results = []

    for doc in docsReview: # Recorremos cada comentario
        similarities = { # Calculamos las distintas similitudes de las distintas palabras claves
            keyWord: doc.similarity(nlp(keyWord))
            for keyWord in keyWords
        }
        
        best_match = max(similarities, key=similarities.get) # Obtenemos el que mejor similitud haya obtenido
        
        results.append({ # Lo guardamos en el array
            "Texto": doc.text,
            "Categoría_predicha": best_match,
            "Similitud": similarities[best_match]
        })
    return results

if __name__ == "__main__":
    # ---------------------------------------------------
    # Apartado 2. El Pipeline Base y Análisis Lingüístico
    #----------------------------------------------------
    try:
        reviewsTxt = load("reviewsLaptop.txt") # Cargamos el txt de los comentarios
    except Exception as e:
        print("No se ha podido cargar el .txt de las reseñas")
        exit()
    pl.Config.set_tbl_rows(-1)  # Muestra todas las filas de las tablas que vayamos a generar en Polars
    # Lo normal, es que los comentarios se guarden por líneas, es decir, un comentario en una línea, otro en otra...
    # Entonces hacemos un splitlines()
    reviews = reviewsTxt.splitlines()
    # Se debe descargar primero con el comando python -m spacy download es_core_news_lg
    nlp = spacy.load("es_core_news_lg") # Cargamos el modelo en español
    docFullReview = nlp(reviewsTxt) # Generamos el Doc con nlp
    cleaningWords = cleaning_words(docFullReview, nlp) # Aplicamos la función del preprocesamiento con el Doc que contiene los comentarios
    print("Las palabras con el preprocesado realizado son: ")
    for i, word in enumerate(cleaningWords, 1):
        print(f"{i}.- {word}")

    # Añadimos aqui la librería de hugging face al pipeline, ya que se tiene que hacer antes
    # de generar los Doc de los distintos comentarios por separados para hacer el análisis de cada uno de ellos
    # Si se hace después de generar los doc de los comentarios, no se incluye en el Pipeline las distintas operaciones
    # En este caso, es una librería que permite hacer analisis de sentimientos de texto en español

    classifier = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
    nlp.add_pipe("hf_sentiment_component", last=True) # Lo añadimos al Pipeline con el nombre del componente de la función creada anteriormente
    Doc.set_extension("hf_sentiment", default=None) # Añadimos las extensiones para obtener los distintos valores que nos devuelve la librería
    Doc.set_extension("hf_score", default=None)

    # Otra cosa importante añadir al Pipeline antes de generar los Doc de cada comentario es el EntityRuler de las entidades que no reconoce
    # En este caso, no reconoce ciertas entidades, sobre todo siglas como E.S.O, GB... entonces hacemos el EntityRule acerca de eso
    # Primero hay que hacer un patrón para reconozca dichas entidades, en este caso es cuando sea todas las letras mayúsculas y tegna una longitud mayor de dos
    patterns = [{
            "label": "SIGLA",
            "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 2}}],
            "id": "sigla_pattern",
            "description": "Sigla formada solo por mayúsculas"
        }]
    pattern_descriptions = {
        "sigla_pattern": "Sigla formada solo por mayúsculas"
    }
    ruler = nlp.add_pipe("entity_ruler", after="ner") # Lo añadimos al Pipeline después de 'ner', 
    # ya que si reconoce una entidad que es una organización, 
    # es preferible que aparezca eso a que aparezca que sean siglas

    ruler.add_patterns(patterns) # Añadimos el patrón que hemos definido
    # Creamos un custom extension para guardar la descripción
    Span.set_extension("description", default=None, force=True) # Añadimos la descripción de la enitdad creada

    docsReview = []
    for comment in reviews: # Obtenemos los distintos Doc de cada comentario
        docReview = nlp(comment)
        docsReview.append(docReview)

    randomNumber = Random().randint(0,len(docsReview)-1) # Generamos un número aleatorio para coger un comentario
    randomSentence = docsReview[randomNumber]
    print(f"El comentario que se ha obtenido aleatoriamente ha sido: {randomSentence}")
    # Obtenemos la etiqueta gramatical de cada palabra
    pos_token = pl.DataFrame([{
        "Texto": token.text,
        "POS": token.pos_ # Mostramos la etiqueta gramatical de cada palabra de la oración
    } for token in randomSentence])
    print(f"A continuación se muestra la etiqueta gramatical de cada palabra del comentario: ")
    print(pos_token)
    print("Guardando árbol de dependencias de la oración anterior...")
    html = displacy.render(randomSentence, style="dep") # Generamos el árbol de dependencia de la oración obtenida
    with open("arbolDependencia-AGG.html", "w", encoding="utf-8") as f: # Lo guardamos como HTML
        f.write(html)
    print("Arbol guardado correctamente, podrás ver el html del árbol de dependencias en el archivo generado")

    # Descargamos el modelo de stanza para generar el árbol de constituyentes
    stanza.download('es', processors='tokenize,pos,constituency,sentiment', verbose=False)
    stanza_nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,constituency,sentiment', use_gpu=False, verbose=False)

    # Árbol de constituyentes con Stanza
    stanza_doc = stanza_nlp(randomSentence.text)
    for i, sentence in enumerate(stanza_doc.sentences):
        print(f"\nÁrbol de constituyentes de la oración {i+1} (si la oración es muy grande, es posible que no quepa en consola):")
        tree = Tree.fromstring(str(sentence.constituency))
        tree.pretty_print() # árbol en consola
    verbs = [token.lemma_ for token in randomSentence if token.pos_ in ("VERB", "AUX")]
    print(f"Los verbos de la oración son los siguientes: {verbs}")

    # APARTADO 3. Extracción de Entidades

    ents_token = extract_entities(docsReview)
    print("A continuación se muestra las entidades reconocidas en los distintos comentarios: ")
    print(ents_token)

    # APARTADO 4. Análisis de Sentimiento

    # Añadimos aqui el modelo de analisis y no antes ya que este modelo de análisis de sentimiento lo utilizaremos solo para los comentarios en ingles
    nlp.add_pipe('spacytextblob')
    nltk.download('vader_lexicon') # Descargamos el modelo VADER de nltk para también hacer el análisis con este modelo
    sia = SentimentIntensityAnalyzer()
    Doc.set_extension("vader_scores", default=None) # Guardamos ambas extensiones para poder usarlo posteriormente
    Doc.set_extension("vader_compound", default=None)
    nlp.add_pipe("vader_sentiment", last=True) # Añadimos el componente creado al Pipeline
    results = sentiment_analysis(docsReview, nlp, stanza_nlp)
    # Crear DataFrame de Polars
    sentiment_df = pl.DataFrame(results)
    print("Ahora se muestran los distintos análisis de sentimientos con diferentes modelos.")
    print("En el spacy-textblob, si el valor es cero, quiere decir que es un comentario neutral, si es mayor que 0 es positivo (cuanto más cerca de 1 más positivo es)")
    print("y si es menor que 0 es negativo")
    print("En cuanto al Stanza, hay que tener en cuenta que hago la media, ya que en un comentario puede haber más de una frase y Stanza")
    print("lo analiza frase por frase, entonces, si es menor que 1 tiende a ser un comentario negativo, si es cercano a 1 es un comentario neutral")
    print("y si es mayor que uno tiende a ser un comentario positivo")
    print("Por otro lado, el VADER de NLTK, si su valor es mayor o igual que 0.5, es un comentario positivo,")
    print("si el valor es entre -0.05 y 0.05 es un comentario neutral, y si es menor que -0.05 es un comentario negativo")
    print("Por último, en cuanto al modelo de Hugging Face, te devuelve un label que te indica si es positivo, neutral o negativo")
    print("y una puntuación que te dice como de intensidad es ese sentimiento")
    print(sentiment_df)

    # APARTADO 5. Descubrimiento del Tópico

    # Estrategia A: Frecuencia de Sustantivos

    chunksDf = extract_noun_chunks(docsReview)
    top_3_chunks = (chunksDf.group_by("Sintagma").len().sort("len", descending=True).head(3)) # Obtenemos los tres sintagmas que más se repiten
    print("A continuación se muestra los tres sintagmas nominales que más se repiten en los comentarios: ")
    print(top_3_chunks)

    pos_nouns = extract_nouns(docsReview)
    top_lemmas = (pos_nouns.group_by("Lemma").len().sort("len", descending=True).head(10)) # Se obtiene los sustantivos que más se repiten para el análisis
    print("Ahora se pueden observar los diez sustantivos que más se repiten: ")
    print(top_lemmas)
    print("Podemos ver que se repiten muchos sustantivos como portátil, ordenador, equipo o pantalla")

    # Estrategia B: Similitud Semántica

    keyWords = ["política", "deportes", "tecnología", "cine", "portátil"] # Definimos las palabras claves para calcular la similitud con el comentario
    processed_reviews = []
    # Recorremos las distintas listas de las palabras preprocesadas
    for docsList in cleaningWords:
        clean_tokens = []
        for commentClean in docsList:
            clean_tokens.append(commentClean) # Lo añadimos a una lista auxiliar
        processed_reviews.append(nlp(" ".join(clean_tokens))) # Añadimos el doc generado con el comentario sin stopwords, signos de puntuación... Ya que podrían generar algo de ruido

    results = get_similarities(processed_reviews, keyWords)
    resultsDf = pl.DataFrame(results) # Creamos un dataframe con esos resultados
    print("Por último, podemos ver con que palabras claves tiene mayor similitud los comentarios: ")
    print(resultsDf)
    # Contar repeticiones de cada categoría
    categoria_mas_frecuente = (resultsDf.group_by("Categoría_predicha").len().sort("len", descending=True).head(1)) # solo la categoría más repetida
    # Extraemos el nombre de la categoría y el número de apariciones
    top_categoria = categoria_mas_frecuente[0, "Categoría_predicha"]
    num_apariciones = categoria_mas_frecuente[0, "len"]
    print(f"La categoría más repetida es: {top_categoria} (aparece {num_apariciones} veces)")