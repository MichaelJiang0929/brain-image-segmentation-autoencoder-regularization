from skimage.transform import resize as resize2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import zoom


def preprocess_label(img, out_shape=(112, 144, 112), mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 3  # GD-enhancing Tumor (ET)
    ncr = resize(ncr, out_shape, mode=mode)
    ed = resize(ed, out_shape, mode=mode)
    et = resize(et, out_shape, mode=mode)
    return np.array([ncr, ed, et], dtype=np.uint8)


def resize(img, out_shape=(112, 144, 112), mode='constant'):
    factors = (
        out_shape[0]/img.shape[0],
        out_shape[1]/img.shape[1],
        out_shape[2]/img.shape[2]
    )
    return zoom(img, factors, mode=mode)


files = sorted(os.listdir(
    './drive/MyDrive/FA20_CS446_Project_Data/data_pub/train'))

train_size = 204
features = np.empty((train_size, 4, 112, 144, 112), dtype=np.float32)
labels = np.empty((train_size, 3, 112, 144, 112), dtype=np.uint8)
i = 0
s = 0
for file in files:
    if file[4:7] == 'img':
        data = np.load(
            './drive/MyDrive/FA20_CS446_Project_Data/data_pub/train/'+file)
        out_shape = (4, 112, 144, 112)
        data = np.array([resize(data[m], (112, 144, 112))
                         for m in [0, 1, 2, 3]], dtype=np.float32)
        data = np.array(data).astype(np.float32)
        features[i] = data
        i += 1
        if i % 50 == 0:
            print(i)
    if file[4:7] == 'seg':
        data = np.load(
            './drive/MyDrive/FA20_CS446_Project_Data/data_pub/train/'+file)
        data = preprocess_label(data)
        labels[s] = data
        s += 1
        if s % 50 == 0:
            print(s)

validation_files = sorted(os.listdir(
    './drive/MyDrive/FA20_CS446_Project_Data/data_pub/validation'))
v_train_size = 68
validation_features = np.empty(
    (v_train_size, 4, 112, 144, 112), dtype=np.float32)
validation_labels = np.empty((v_train_size, 3, 112, 144, 112), dtype=np.uint8)
i = 0
s = 0
for file in validation_files:
    if file[4:7] == 'img':
        data = np.load(
            './drive/MyDrive/FA20_CS446_Project_Data/data_pub/validation/'+file)
        out_shape = (4, 112, 144, 112)
        data = np.array([resize(data[m], (112, 144, 112))
                         for m in [0, 1, 2, 3]], dtype=np.float32)
        data = np.array(data).astype(np.float32)
        validation_features[i] = data
        i += 1
        if i % 50 == 0:
            print(i)
    if file[4:7] == 'seg':
        data = np.load(
            './drive/MyDrive/FA20_CS446_Project_Data/data_pub/validation/'+file)
        data = preprocess_label(data)
        validation_labels[s] = data
        s += 1
        if s % 50 == 0:
            print(s)

input_shape = (4, 112, 144, 112)
with tf.device('/gpu:1'):
    tf.compat.v1.disable_eager_execution()
    model = build_model(input_shape=input_shape, output_channels=3)
    model.fit(features, labels, batch_size=1, epochs=200)

test_files = sorted(os.listdir(
    './drive/MyDrive/FA20_CS446_Project_Data/test_pub/test_pub'))


def predict(result):
    matrix = np.zeros((112, 144, 112))
    for i in range(112):
        for j in range(144):
            for k in range(112):
                # find the maximum position in 3 channels
                values = np.array(
                    [result[0][i][j][k], result[1][i][j][k], result[2][i][j][k]])
                max_ind = np.argmax(values)
                if values[max_ind] >= 0.5:
                    matrix[i, j, k] = max_ind+1
                else:
                    matrix[i, j, k] = 0.0

    return matrix


def padding_image(H, W, D, predictions):
    HL = (H - 112) // 2
    HR = (H - HL - 112)
    WL = (W - 144) // 2
    WR = (W - WL - 144)
    DL = (D - 112) // 2
    DR = (D - DL - 112)
    predictions = np.pad(predictions, pad_width=(
        (HL, HR), (WL, WR), (DL, DR)), mode='constant', constant_values=0)
    return predictions


def getandsavepredict(index):
    data = np.load(
        './drive/MyDrive/FA20_CS446_Project_Data/test_pub/test_pub/' + index + '_imgs.npy')
    channel, H, W, D = data.shape
    out_shape = (4, 112, 144, 112)
    data = resize2(data, out_shape)
    data = np.array(data).astype(np.float32)
    predictions = model.predict(data.reshape((1, 4, 112, 144, 112)))[0]
    res = predict(predictions)
    res = padding_image(H, W, D, res)
    np.save('./drive/MyDrive/FA20_CS446_Project_Data/test_pub/test_pub/' +
            index + '_seg.npy', res)
