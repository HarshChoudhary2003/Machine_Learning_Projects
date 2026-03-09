import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, hamming_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalAveragePooling1D, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import setup_logging, save_plot, clean_text

logger = setup_logging()

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

class EmotionClassifier:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.embedding_matrix = None
        self.history = {}

    def prepare_data(self, texts, labels, multi_label=False):
        """
        Prepares sequences and performs train/val/test split.
        - stratified split (70/15/15)
        """
        logger.info("Preprocessing GoEmotions dataset...")
        cleaned_texts = [clean_text(t) for t in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = np.array(labels)

        # Split: 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if not multi_label else None)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if not multi_label else None)
        
        logger.info(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_dense_model(self, num_classes, multi_label=False):
        """Embedding -> GlobalAveragePooling -> Dense -> Dropout -> Dense -> Output"""
        input_dim = self.embedding_matrix.shape[0] if self.embedding_matrix is not None else self.max_words
        model = Sequential([
            Embedding(input_dim=input_dim, output_dim=self.embedding_dim, weights=[self.embedding_matrix] if self.embedding_matrix is not None else None, input_length=self.max_len, trainable=False if self.embedding_matrix is not None else True),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='sigmoid' if multi_label else 'softmax')
        ])
        return model

    def build_cnn_model(self, num_classes, multi_label=False):
        """Embedding -> Conv1D -> GlobalMaxPooling -> Dense -> Output"""
        input_dim = self.embedding_matrix.shape[0] if self.embedding_matrix is not None else self.max_words
        model = Sequential([
            Embedding(input_dim=input_dim, output_dim=self.embedding_dim, weights=[self.embedding_matrix] if self.embedding_matrix is not None else None, input_length=self.max_len, trainable=False if self.embedding_matrix is not None else True),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='sigmoid' if multi_label else 'softmax')
        ])
        return model

    def build_bilstm_model(self, num_classes, multi_label=False, use_attention=True):
        """Embedding -> Bidirectional LSTM -> (Attention) -> Dense -> Output"""
        input_dim = self.embedding_matrix.shape[0] if self.embedding_matrix is not None else self.max_words
        
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        
        inputs = Input(shape=(self.max_len,))
        x = Embedding(input_dim=input_dim, output_dim=self.embedding_dim, weights=[self.embedding_matrix] if self.embedding_matrix is not None else None, input_length=self.max_len, trainable=False if self.embedding_matrix is not None else True)(inputs)
        
        if use_attention:
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            x = Attention()(x)
        else:
            x = Bidirectional(LSTM(64))(x)
            
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='sigmoid' if multi_label else 'softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32, multi_label=False):
        """Trains the model with early stopping and class weights."""
        loss = 'binary_crossentropy' if multi_label else 'sparse_categorical_crossentropy'
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        
        # Calculate class weights for imbalance (if single label)
        class_weight_dict = None
        if not multi_label:
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(weights))

        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        return history

    def plot_history(self, history, model_name='model'):
        """Plots training loss and accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['loss'], label='Train')
        ax1.plot(history.history['val_loss'], label='Val')
        ax1.set_title(f'{model_name} Loss')
        ax1.legend()
        
        ax2.plot(history.history['accuracy'], label='Train')
        ax2.plot(history.history['val_accuracy'], label='Val')
        ax2.set_title(f'{model_name} Accuracy')
        ax2.legend()
        
        save_plot(fig, f'{model_name}_learning_curves.png')

    def evaluate(self, model, X_test, y_test, label_names=None, multi_label=False):
        """Evaluates the model and calculates detailed metrics."""
        probs = model.predict(X_test)
        if multi_label:
            preds = (probs > 0.5).astype(int)
        else:
            preds = np.argmax(probs, axis=1)
            
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro')
        h_loss = hamming_loss(y_test, preds)
        
        logger.info(f"Evaluation Results - Accuracy: {acc:.4f}, F1 Macro: {f1_macro:.4f}, Hamming Loss: {h_loss:.4f}")
        
        if not multi_label:
            # Confusion Matrix Heatmap
            plt.figure(figsize=(15, 12))
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=label_names, yticklabels=label_names)
            plt.title("Confusion Matrix Heatmap")
            save_plot(plt.gcf(), "cm_neural_heatmap.png")
            
        print(classification_report(y_test, preds, target_names=label_names))
        return preds, probs

    def predict_single(self, model, text, label_names, multi_label=False):
        """Predicts emotion for a single piece of text."""
        cleaned = clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_len)
        probs = model.predict(padded)[0]
        
        if multi_label:
            results = {label_names[i]: float(probs[i]) for i in range(len(label_names)) if probs[i] > 0.5}
            if not results: # If none above 0.5, return top 1
                top_idx = np.argmax(probs)
                results = {label_names[top_idx]: float(probs[top_idx])}
        else:
            top_idx = np.argmax(probs)
            results = {label_names[top_idx]: float(probs[top_idx])}
            
        return results

