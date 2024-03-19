import pickle
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add, Dense, Dropout, LSTM, Flatten, Input
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, Activation
from tensorflow.python.keras.models import Input, Model, load_model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from scipy.interpolate import splev, splrep
import pandas as pd

base_dir = "dataset"

ir = 3 # interpolate interval
before = 2
after = 2

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test


# Define the positional encoding function
def positional_encoding(seq_length, d_model):
    position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.pow(10000.0, 2.0 * tf.range(d_model // 2, dtype=tf.float32) / d_model)
    angle = tf.matmul(position, div_term[tf.newaxis, :])
    sin = tf.sin(angle)
    cos = tf.cos(angle)
    pos_encoding = tf.concat([sin, cos], axis=-1)
    return pos_encoding

def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
    # Layer Normalization
    normalized_input = LayerNormalization()(inputs)

    seq_length = tf.shape(inputs)[1]  # Get the sequence length dynamically
    pos_enc = positional_encoding(seq_length, d_model=128)
    transformer_input_with_pos = normalized_input + pos_enc

    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(transformer_input_with_pos, transformer_input_with_pos)

    # Add & Normalize
    attention_output = Add()([transformer_input_with_pos, attention_output])
    normalized_output = LayerNormalization()(attention_output)

    # Feed Forward Network
    ff_output = Dense(128, activation='relu')(normalized_output)
    ff_output = Dense(128)(ff_output)

    # Add & Normalize
    encoder_output = Add()([normalized_output, ff_output])
    normalized_encoder_output = LayerNormalization()(encoder_output)

    # Dropout and Flatten
    dropout_output = Dropout(dropout_rate)(normalized_encoder_output)

    return dropout_output

def create_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # CNN block
    x = Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)

    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)

    cnn_output = Dropout(0.5)(x)

    # Transformer Encoder Block
    transformer_output = transformer_encoder_block(cnn_output, num_heads=2, key_dim=32, dropout_rate=0.5)

    # LSTM
    lstm_output = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(transformer_output)

    # Fully Connected Layers
    fc_output = Flatten()(lstm_output)
    fc_output = Dense(128, activation='relu')(fc_output)
    outputs = Dense(2, activation="softmax")(fc_output)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], "r-", label="Training Loss", linewidth=0.5)
    axes[0].plot(history["val_loss"], "b-", label="Validation Loss", linewidth=0.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], "r-", label="Training Accuracy", linewidth=0.5)
    axes[1].plot(history["val_accuracy"], "b-", label="Validation Accuracy", linewidth=0.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig('hist.png')
    plt.show()

    # save prediction score
    y_score = model.predict(x_test)
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_score[:, 1], "subject": groups_test})
    output.to_csv("CNN-Transformer-LSTM.csv", index=False)
    y_true, y_pred = np.argmax(y_test, axis=-1), np.argmax(model.predict(x_test, batch_size=1024, verbose=1), axis=-1)
    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_true, y_pred, average='binary')
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("acc: {}, sn: {}, sp: {}, f1: {}, AUC: {}".format(acc, sn, sp, f1, roc_auc))

    # Define labels for confusion matrix
    labels = ['Non-Apnea', 'Apnea']

    # Calculate confusion matrix
    C = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(C, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('Confusion Matrix.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('ROC Curve.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print("train num:", len(y_train))
    print("test num:", len(y_test))

    model = create_model(input_shape=x_train.shape[1:])
    model.summary()

    plot_model(model, "model.png", show_shapes=True)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model', monitor='val_loss', verbose=1,
                                                    save_best_only=True)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
    redonplat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger('log.csv', separator=',', append=True)

    callbacks_list = [
        checkpoint,
        early,
        redonplat,
        csv_logger,
    ]

    history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test), callbacks=callbacks_list)
    model.save(os.path.join("model.final.h5"))

    plot(history.history)
