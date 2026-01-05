import pickle
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"
from numba import cuda

device = cuda.get_current_device()
device.reset()
import tensorflow as tf

a = tf.zeros([], tf.float32)
from keras import backend as K
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add, Dense, Dropout, LSTM, Flatten, Input, \
    Lambda
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, Activation, \
    Bidirectional, GRU, SimpleRNN
from tensorflow.python.keras.models import Input, Model, load_model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import KFold
from scipy.interpolate import splev, splrep
import pandas as pd

base_dir = "dataset"

ir = 3  # interpolate interval
before = 2
after = 2

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:  # read preprocessing result
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
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))  # convert to numpy format
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
    x = MaxPooling1D(pool_size=4, strides=4)(x)

    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)

    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4, strides=4)(x)

    cnn_output = Dropout(0.5)(x)

    # Transformer Encoder Block
    transformer_output = transformer_encoder_block(cnn_output, num_heads=2, key_dim=32, dropout_rate=0.5)

    # LSTM
    lstm_output = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(transformer_output)

    # Fully Connected Layers after LSTM
    fc_output = Flatten()(lstm_output)
    #fc_output = Dropout(0.5)(fc_output)
    fc_output = Dense(128, activation='relu')(fc_output)
    outputs = Dense(2, activation="softmax")(fc_output)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


def evaluate_model(model, x_val, y_val, groups_val, fold):
    # save prediction score
    y_score = model.predict(x_val)

    # Create DataFrame without subject information if groups_val is not available
    try:
        output = pd.DataFrame({
            "y_true": y_val[:, 1],
            "y_score": y_score[:, 1],
            "subject": groups_val
        })
    except:
        output = pd.DataFrame({
            "y_true": y_val[:, 1],
            "y_score": y_score[:, 1]
        })
    output.to_csv(f"CNN-Transformer-LSTM-fold-{fold}.csv", index=False)

    y_true, y_pred = np.argmax(y_val, axis=-1), np.argmax(model.predict(x_val, batch_size=1024, verbose=1), axis=-1)
    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    po = (TP + TN) / (TP + TN + FP + FN)
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + TN + FP + FN) ** 2
    kappa = (po - pe) / (1 - pe)
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_true, y_pred, average='binary')
    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_val[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    print(f"\nFold {fold} Results:")
    print("acc: {}, sn: {}, sp: {}, f1: {}, AUC: {}, kappa: {}".format(acc, sn, sp, f1, roc_auc, kappa))

    return acc, sn, sp, f1, roc_auc, kappa

def plot(history):
    sns.set(style="white")  # Apply seaborn style

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    sns.lineplot(ax=axes[0], data=history["loss"], label="Training Loss", color="red", linewidth=1)
    sns.lineplot(ax=axes[0], data=history["val_loss"], label="Validation Loss", color="blue", linewidth=1)
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy Plot
    sns.lineplot(ax=axes[1], data=history["accuracy"], label="Training Accuracy", color="red", linewidth=1)
    sns.lineplot(ax=axes[1], data=history["val_accuracy"], label="Validation Accuracy", color="blue", linewidth=1)
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("hist.png", bbox_inches="tight", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load data
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

    # Convert to categorical
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print("train num:", len(y_train))
    print("test num:", len(y_test))

    # Set up 5-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize lists to store results
    fold_results = []
    best_model = None
    best_accuracy = 0

    # Merge train and test data for cross-validation
    x_full = np.concatenate((x_train, x_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    groups_full = np.concatenate((groups_train, groups_test), axis=0)

    # Start 5-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(x_full, y_full), 1):
        print(f"\nStarting Fold {fold}/5")

        # Get train and validation data for this fold
        x_train_fold, y_train_fold = x_full[train_idx], y_full[train_idx]
        x_val_fold, y_val_fold = x_full[val_idx], y_full[val_idx]
        groups_val_fold = groups_full[val_idx]

        # Create model
        model = create_model(input_shape=x_train.shape[1:])
        model.summary()

        # Compile model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

        # Define callbacks
        callbacks_list = [
            tf.keras.callbacks.ModelCheckpoint(filepath=f'model_fold_{fold}.h5', monitor='val_loss',
                                               save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1),
            tf.keras.callbacks.CSVLogger(f'log_fold_{fold}.csv', append=True)
        ]

        # Train model
        history = model.fit(
            x_train_fold,
            y_train_fold,
            batch_size=128,
            epochs=100,
            validation_data=(x_val_fold, y_val_fold),
            callbacks=callbacks_list,
            verbose=1
        )

        # Evaluate model
        acc, sn, sp, f1, roc_auc, kappa = evaluate_model(model, x_val_fold, y_val_fold, groups_val_fold, fold)
        fold_results.append({
            'fold': fold,
            'accuracy': acc,
            'sensitivity': sn,
            'specificity': sp,
            'f1_score': f1,
            'auc': roc_auc,
            'kappa': kappa
        })

        # Track best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

        # Stop after 5 folds
        if fold == 5:
            break

    # Print summary of all folds
    print("\n5-Fold Cross Validation Results Summary:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Accuracy={result['accuracy']:.4f}, Sensitivity={result['sensitivity']:.4f}, "
              f"Specificity={result['specificity']:.4f}, F1={result['f1_score']:.4f}, "
              f"AUC={result['auc']:.4f}, Kappa={result['kappa']:.4f}")

    # Calculate and print average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'sensitivity': np.mean([r['sensitivity'] for r in fold_results]),
        'specificity': np.mean([r['specificity'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results]),
        'kappa': np.mean([r['kappa'] for r in fold_results])
    }

    print("\nAverage Metrics Across All Folds:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {avg_metrics['sensitivity']:.4f}")
    print(f"Specificity: {avg_metrics['specificity']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    print(f"AUC: {avg_metrics['auc']:.4f}")
    print(f"Kappa: {avg_metrics['kappa']:.4f}")

    # Save final log.csv
    log_df = pd.DataFrame(fold_results)
    avg_row = {'fold': 'average'}
    avg_row.update(avg_metrics)
    log_df = pd.concat([log_df, pd.DataFrame([avg_row])], ignore_index=True)
    log_df.to_csv("log.csv", index=False)

    plot(history.history)

    y_score = best_model.predict(x_test)
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_score[:, 1], "subject": groups_test})
    output.to_csv("CNN-Transformer-LSTM.csv", index=False)
    y_true, y_pred = np.argmax(y_test, axis=-1), np.argmax(model.predict(x_test, batch_size=1024, verbose=1), axis=-1)

    # Plot final confusion matrix
    test_cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Apnea', 'Apnea']
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(test_cm, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('Confusion_Matrix_Final.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Save the best model
    if best_model is not None:
        best_model.save("best_model.h5")
        print("\nBest model saved as 'best_model.h5'")