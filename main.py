# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#
# ## 1. Setup
# El kernel del meu notebook, que no esta al colab, te la versió 3.9.0 de Keras y 2.19.0 de Tensorflow

# +
# We set the backend to TensorFlow. The code works with
# both `tensorflow` and `torch`. It does not work with JAX
# due to the behavior of `jax.numpy.tile` in a jit scope
# (used in `causal_attention_mask():`) `tile` in JAX does
# not support a dynamic `reps` argument.
# You can make the code work in JAX by wrapping the
# inside of the `causal_attention_mask` function in
# a decorator to prevent jit compilation:
# `with jax.ensure_compile_time_eval():`.
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import random
import string

import keras
import numpy as np
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras import layers, ops

TextVectorization = layers.TextVectorization
# -

#
# ## 2. Implementació d'un bloc transformer com a capa.


# +
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


# -

# **Què significa que la màscara d'atenció sigui causal?**
#
# Extret del comentari de la funció, bloqueja que cada token pugui veure tokens posteriors i asegura que les prediccions
# només depenguin de tokens anteriors

# ## 3. Implementació de les capes d'embedding.

# Create two separate embedding layers: one for tokens and one for token index (positions).
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# ## 4. Implementació del GPT en miniatura.
#

# +
vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model


# -

# **Quines funcions implementades als punts anteriors s'invoquen ara?**
# * TokenAndPositionEmbedding (Capa d'embedding)
# * TransformerBlock (Bloc transformer com a capa)
#

# ## 5. Dades per al model de llenguatge a nivell de paraula.

# Download the IMDB dataset and combine training and validation sets for a text generation task
# !curl -O data/aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf data/aclImdb_v1.tar.gz data/

# +
batch_size = 128

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
filenames = []
directories = [
    "data/aclImdb/train/pos",
    "data/aclImdb/train/neg",
    "data/aclImdb/test/pos",
    "data/aclImdb/test/neg",
]
for dir in directories:
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# Create a dataset from text files
random.shuffle(filenames)
text_ds = tf_data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tensorflow.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
text_ds = text_ds.prefetch(tf_data.AUTOTUNE)


# -

# **Quina és l'aplicació habitual del dataset Imdb?**
#
# Segons el README.md del zip descarregat, el dataset es pot utilitzar per entrenar models per classificar un comentari en un sentiment positiu o negatiu al que es comenta.

# ## 6. Implementació del callback Keras per generar text.

# +
class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_token_list = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_token_list)
            sample_index = len(start_token_list) - 1
            if pad_len < 0:
                x = start_token_list[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_token_list + [0] * pad_len
            else:
                x = start_token_list
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_token_list.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
total_tokens_generated = 40
text_gen_callback = TextGenerator(total_tokens_generated, start_tokens, vocab)


# -

# **Quin canvi faríeu al codi perquè sempre triàs la paraula més probable?**
#
# Sembla que a `sample_from`, quan es tria el token, ho fa de forma aleatoria amb `np.random.choice(indices, p=preds)`. Si `logits` representa un llistat de tokens amb la seva probabilitat, retornaria el valor màxim d'aquest llistat.

def sample_from(self, logits):
    return np.argmax(logits)


# ## 7. Entrenau el model amb un altre dataset de text d'una mida suficient.

# He escollit un dataset de ressenyes de productes dAmazon. A Kaggle hi ha força datasets disponibles, però són tots massa pesats, oferint molts més textos dels que necessitem (en la magnitud del milió de reviews per arxiu).
#
# He trobat [una web](https://amazon-reviews-2023.github.io/#grouped-by-category) on s'ofereix ressenyes en format `.jsonl`, subcategoritzades, reduint així el nombre de ressenyes a unes cent mil (les més lleugeres).

# !curl -O data/Magazine_Subscriptions.jsonl.gz https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz
# !tar -xf data/Magazine_Subscriptions.jsonl.gz data/Magazine_Subscriptions.jsonl

# +
import pandas as pd


# Carrega les dades des del DataFrame de Pandas
def load_texts_from_dataframe(df, text_column):
    """Extracts texts from a pandas DataFrame column."""
    try:
        texts = df[text_column].astype(str).tolist()  # Ensure texts are strings
        return texts
    except KeyError:
        raise KeyError(f"Column '{text_column}' not found in DataFrame.")


# Carrega les dades des del DataFrame
jsonObj = pd.read_json(path_or_buf="data/Magazine_Subscriptions.jsonl", lines=True)
try:
    texts = load_texts_from_dataframe(jsonObj, "text")
except KeyError as e:
    print(f"Error: {e}")
    exit()


# Crea un dataset de TensorFlow a partir dels textos
text_ds = tf_data.Dataset.from_tensor_slices(texts)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)


def custom_standardization(input_string):
    """Remove html line-break tags and handle punctuation"""
    lowercased = tf_strings.lower(input_string)
    stripped_html = tf_strings.regex_replace(lowercased, "<br />", " ")
    return tf_strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \\1")


# Crea una capa de vectorització i adapta-la al text
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()  # Per recuperar paraules a partir dels índexs


def prepare_lm_inputs_labels(text):
    """
    Desplaça les seqüències de paraules per 1 posició, de manera que l'objectiu per a la posició (i) sigui
    la paraula a la posició (i+1). El model utilitzarà totes les paraules fins a la posició (i)
    per predir la paraula següent.
    """
    text = tensorflow.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels, num_parallel_calls=tf_data.AUTOTUNE)
text_ds = text_ds.prefetch(tf_data.AUTOTUNE)
# -

model = create_model()
# Hem de redefinir els tokens d'entrada, pero que sigués coherent amb les dades
start_prompt = "this magazine is"
start_tokens_list = [word_to_index.get(_, 1) for _ in start_prompt.split()]
total_tokens_generated = 40
text_gen_callback = TextGenerator(total_tokens_generated, start_tokens_list, vocab)

# epochs = 25
epochs = 1  # No tením tot el dia
# model.fit(text_ds, verbose=2, epochs=epochs, callbacks=[text_gen_callback])

# ## 8. Canviau el codi de generació de text, de forma que en lloc d'aturar quan ha generat un nombre de tokens, aturi quan genera un punt. D'aquesta forma, les frases generades sempre seran completes.

# +
class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self,
        max_tokens,
        start_tokens,
        interrupt_token,
        index_to_word,
        top_k=10,
        print_every=1,
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.interrupt_token = interrupt_token
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_token_list = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        # Si troba un . s'atura el while.
        # Encara que s'hauria d'aturar quan trobi un punt, no podemo saber si ho generara, llavors mantenim tambe la
        # condició anterior.
        while (
            self.interrupt_token not in tokens_generated
            and num_tokens_generated <= self.max_tokens
        ):
            pad_len = maxlen - len(start_token_list)
            sample_index = len(start_token_list) - 1
            if pad_len < 0:
                x = start_token_list[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_token_list + [0] * pad_len
            else:
                x = start_token_list
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_token_list.append(sample_token)
            num_tokens_generated += len(tokens_generated)
            print(f"Generated {str(num_tokens_generated)} tokens")

        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this magazine is"
period_token = word_to_index.get(".", 1)
start_tokens_list = [word_to_index.get(_, 1) for _ in start_prompt.split()]
total_tokens_generated = 10000
text_gen_callback = TextGenerator(
    total_tokens_generated, start_tokens_list, period_token, vocab
)
# -

# epochs = 25
epochs = 1
model.fit(text_ds, verbose=1, epochs=epochs, callbacks=[text_gen_callback])

# No troba cap punt, potser sigui perquè quan fem servir les dades, a `custom_standardization`, li traiem tots els punts als textos de les ressenyes.

# ## 9. Comparau el rànquing de xatbots disponible a lmarena.ai amb el dels apunts. Quines diferències hi destacau?
#
# Als apunts no s'esmenta grok, que actualment ocupa el primer lloc al rànquing. Pel que fa a la resta de chatbots, sembla que dels esmentats als apunts, Gemini, ChatGPT i Claude mantenen les seves posicions. A destacar Crida que ha augmentat considerablement la seva posició.
# El rànquing dels apunts estava dominat per models de Google i OpenAI, però a l'actual, està més diversificat, incloent-hi fins i tot models amb llicències d'ús obert.

# ## 10. Provau alguns LLM que hi ha disponibles a través de la interfície de xat de HuggingFace i comentau les diferències que hi heu observat. Hi ha models recents com DeepSeek 3 i Grok 3?
#
#

# He demanat el mateix prompt als models:
#
# **Si vull anar de Palma de Mallorca fins a Hyderabad amb cotxe, explica els requisits que necessito complir, com a documentació necessària, vacunes, zones a evitar i punts d'interès pels quals valguin la pena desviar-se del camí més curt i ràpid.**
#
# He desat les respostes al [directori de resposts d'aquest repo](respostes_models/README.md.md)
#
#
# ### Llama
# Resposta balanceada y completa.  Fa mencio a tot les peticions fetas.
#
# ### Qwen
# Resposta molt parescuda a la de Llama, pero te errors, com el llistat de paisos a visitar i el visats necessaris per viatjar a aquests països.
#
# ### Deepkseek
# A meitat de resposta, el model comença a barrejar el text amb les etiquetes de markdown i l'idioma de la resposta amb altres.
#
# ### Mistral
# Resposta molt curta i incompleta. Mistral falla a esmentar aspectes claus del llistat de peticions
#
#
# **Hi ha models recents com DeepSeek 3 i Grok 3?**
#
# Actualment Deepseek V3 no està disponible a la web, però si el model Deepseek R1, que és més modern, no es troba entre la llista de models disponibles.
