import spacy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import re
import nltk
import spacy
from transformers import BertTokenizer, TFAutoModel
import string

class NLP:
    def __init__(self, model_bert, model_spacy):
        self.seq_len = 512
        self.bert_model = self.load_bert(model_bert)
        self.spacy_model = spacy.load(model_spacy)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        print("models loaded")

    def strip_links(self, text):
        link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links         = re.findall(link_regex, text)
        for link in links:
            text = text.replace(link[0], ', ')    
        return text

    def strip_all_entities(self, text):
        entity_prefixes = ['@','#']
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)

    def preprocess(self, text):
        text = text.lower()
        text = self.strip_all_entities(self.strip_links(text))
        text = nltk.word_tokenize(text)
        text = " ".join([word for word in text if word not in nltk.corpus.stopwords.words('english')])

        return text
    
    def preprocess_spacy(self, text):
        text = text.lower()
        text = self.strip_all_entities(self.strip_links(text))
    
        return text

    def load_bert(self, weights_path):
        bert = TFAutoModel.from_pretrained('bert-base-cased')
        input_ids = keras.layers.Input(shape=(self.seq_len,), name="input_ids", dtype="int32")
        attention_mask = keras.layers.Input(shape=(self.seq_len,), name="attention_mask", dtype="int32")

        embeddings = bert.bert(input_ids, attention_mask=attention_mask)[1]
        
        x = layers.Dense(1024, activation="relu")(embeddings)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=[input_ids, attention_mask], outputs=x)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.load_weights(weights_path)
        return model

    def get_bert_embedding(self, texts):
        num_samples = len(texts)

        Xids = np.zeros((num_samples, self.seq_len))
        Xmask = np.zeros((num_samples, self.seq_len))
    
        for i, phrase in enumerate(texts):
            token = self.bert_tokenizer.encode_plus(
            phrase, max_length=self.seq_len, add_special_tokens=True, 
            padding="max_length", truncation=True, return_tensors='tf')

        Xids[i, :] = token['input_ids']
        Xmask[i, :] = token['attention_mask']

        return Xids, Xmask
    
    def process_bert(self, texts):
        assert type(texts) == list
        texts = [self.preprocess(text) for text in texts]

        Xids, Xmask = self.get_bert_embedding(texts)
        labels = self.bert_model.predict([Xids, Xmask])

        return labels

    def process_spacy(self, texts):
        assert type(texts) == list
        texts = [self.preprocess_spacy(text) for text in texts]

        labels = []
        for text in texts:
            doc = self.spacy_model(text)
            labels.append(doc)

        return labels