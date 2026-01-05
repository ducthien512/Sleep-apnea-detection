import os
# Ensure TensorFlow uses GPU efficiently
os.environ["TF_GPU_ALLOCATOR"] = "default"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

from mat4py import loadmat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Add, Dense, Dropout, LSTM, Flatten, Input, Lambda, Conv1D, MaxPooling1D, BatchNormalization, Activation
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from scipy.interpolate import splev, splrep

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Print TensorFlow device placement
print("TensorFlow is built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Using Device:", tf.test.gpu_device_name())

list_string = ['ucddb002', 'ucddb003', 'ucddb005', 'ucddb006', 'ucddb007', 'ucddb009', \
               'ucddb010', 'ucddb012', 'ucddb014', 'ucddb015', 'ucddb017', \
               'ucddb019', 'ucddb020', 'ucddb021', 'ucddb022', 'ucddb023', 'ucddb024', \
               'ucddb025', 'ucddb026', 'ucddb027', 'ucddb028']  # 'ucddb008','ucddb011','ucddb013','ucddb018',

valid_features = np.zeros((1, 1408))
test_features = np.zeros((1, 1408))
valid_labels = np.zeros((1, 1))
test_labels = np.zeros((1, 1))

for l in list_string:
    ecg_valid = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_ecg_valid.mat')
    ecg_valid = np.array(ecg_valid['ecg_valid'])
    ecg_valid_labels = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_valid_labels.mat')
    ecg_valid_labels = np.array(ecg_valid_labels['class_valid'])
    valid_features = np.append(valid_features, ecg_valid, axis=0)
    valid_labels = np.append(valid_labels, ecg_valid_labels)

    ecg_test = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_ecg_test.mat')
    ecg_test = np.array(ecg_test['ecg_test'])
    ecg_test_labels = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_test_labels.mat')
    ecg_test_labels = np.array(ecg_test_labels['class_test'])
    test_features = np.append(test_features, ecg_test, axis=0)
    test_labels = np.append(test_labels, ecg_test_labels)

ecg_valid = valid_features[1:, :]
# ecg_valid=np.expand_dims(ecg_valid, axis=2)
valid_labels = valid_labels[1:]
valid_labels = valid_labels.flatten()

ecg_test = test_features[1:, :]
# ecg_test=np.expand_dims(ecg_test, axis=2)
test_labels = test_labels[1:]
test_labels = test_labels.flatten()

train_features = np.zeros((1, 1408))

train_labels = np.zeros((1, 1))

for l in list_string:
    ecg_train = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_ecg_train.mat')
    ecg_train = np.array(ecg_train['ecg_train'])
    ecg_train_labels = loadmat('D:\\PhD\\Transformer Neural Network\\Sleep Apnea detection\\UCDDB\\' + l + '_train_labels.mat')
    ecg_train_labels = np.array(ecg_train_labels['class_train'])
    train_features = np.append(train_features, ecg_train, axis=0)
    train_labels = np.append(train_labels, ecg_train_labels)

ecg_train = train_features[1:, :]
# ecg_train=np.expand_dims(ecg_train, axis=2)
train_labels = train_labels[1:]
train_labels = train_labels.flatten()

ecg_train_mean = np.mean(ecg_train)
ecg_train_std = np.std(ecg_train)

for i in range(ecg_train.shape[0]):
    ecg_train[i, :] = (ecg_train[i, :] - ecg_train_mean) / ecg_train_std

for i in range(ecg_valid.shape[0]):
    ecg_valid[i, :] = (ecg_valid[i, :] - ecg_train_mean) / ecg_train_std

for i in range(ecg_test.shape[0]):
    ecg_test[i, :] = (ecg_test[i, :] - ecg_train_mean) / ecg_train_std

# ecg_train=(ecg_train-ecg_train_mean)/ecg_train_std
# ecg_valid=(ecg_valid-ecg_train_mean)/ecg_train_std
# ecg_test=(ecg_test-ecg_train_mean)/ecg_train_std

ecg_train = np.expand_dims(ecg_train, axis=2)
ecg_valid = np.expand_dims(ecg_valid, axis=2)
ecg_test = np.expand_dims(ecg_test, axis=2)

y_train = train_labels
y_valid = valid_labels
y_test = test_labels

keras.backend.clear_session()

num_classes = 2

y_train_encoder = sklearn.preprocessing.LabelEncoder()
y_train_num = y_train_encoder.fit_transform(y_train)
y_train_wide = keras.utils.to_categorical(y_train_num, num_classes)

y_valid_num = y_train_encoder.fit_transform(y_valid)
y_valid_wide = keras.utils.to_categorical(y_valid_num, num_classes)

y_test_num = y_train_encoder.fit_transform(y_test)
y_test_wide = keras.utils.to_categorical(y_test_num, num_classes)


# Define CNN-Transformer-LSTM model
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

#CNN-Transformer-LSTM

inputs = keras.layers.Input(shape=(1408, 1))

# Batch Normalization
#x = BatchNormalization()(inputs)

# CNN Block
x = Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
x = MaxPooling1D(pool_size=4, strides=4)(x)

x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
x = MaxPooling1D(pool_size=4, strides=4)(x)
x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
x = MaxPooling1D(pool_size=4, strides=4)(x)
x = Dropout(0.5)(x)

# Transformer Block
x = transformer_encoder_block(x, num_heads=2, key_dim=32, dropout_rate=0.5)

# LSTM Block
x = LSTM(units=128, dropout=0.5, activation='relu', return_sequences=True)(x)

# Fully Connected Layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation="softmax")(x)

# Create model
model_ecg = keras.models.Model(inputs, outputs)

# Print model summary
model_ecg.summary()
plot_model(model_ecg, "model.png", show_shapes=True)  # Plot model

model_ecg.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

batch_size = 128
epochs = 100

print("ECG train shape:", ecg_train.shape)

best_weights_filepath_ecg = './best_weights_ecg_cnn_transformer_lstm.weights.h5'

mcp_ecg = keras.callbacks.ModelCheckpoint(best_weights_filepath_ecg, monitor="val_accuracy",
                          save_best_only=True, save_weights_only=True)

history = model_ecg.fit(ecg_train, y_train_wide,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(ecg_valid, y_valid_wide),
                        callbacks=[mcp_ecg])

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8,6))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss, 'red', label='Training Loss')
plt.plot(val_loss, 'blue', label='Validation Loss')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Loss vs Epochs')
plt.legend()
plt.savefig('Loss.png', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(8,6))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(acc, 'red', label='Training Accuracy')
plt.plot(val_acc, 'blue', label='Validation Accuracy')
plt.xticks(range(0,epochs)[0::100])
plt.title('Training and Validation Accuracy vs Epochs')
plt.legend()
plt.savefig('Accuracy.png', bbox_inches='tight', dpi=300)
plt.show()

# Load best model weights
model_ecg.load_weights('./best_weights_ecg_cnn_transformer_lstm.weights.h5')

# Predict probabilities
y_pred = model_ecg.predict(ecg_test)
predict_test = np.argmax(y_pred, axis=1)
predict_test = predict_test.reshape(predict_test.shape[0], 1)

# Compute confusion matrix
cm = confusion_matrix(y_test_num, predict_test)

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Compute Cohen's Kappa
po = (TP + TN) / (TP + TN + FP + FN)
pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + TN + FP + FN) ** 2
kappa = (po - pe) / (1 - pe)

# Compute Accuracy, Sensitivity, Specificity
acc = (TP + TN) / (TP + TN + FP + FN)
sn = TP / (TP + FN)  # Sensitivity (Recall)
sp = TN / (TN + FP)  # Specificity

# **Fix F1-score Calculation**
f1 = f1_score(y_test_num, predict_test, average='binary')

# **Fix AUC Calculation**
fpr, tpr, _ = roc_curve(y_test_wide[:, 1], y_pred[:, 1])  # Use softmax output
roc_auc = auc(fpr, tpr)

print("acc: {:.4f}, sn: {:.4f}, sp: {:.4f}, f1: {:.4f}, AUC: {:.4f}, kappa: {:.4f}".format(acc, sn, sp, f1, roc_auc, kappa))

# Define labels for confusion matrix
labels = ['Non-Apnea', 'Apnea']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png', bbox_inches='tight', dpi=300)
plt.show()


