import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DATA_FILE = 'train_sensor_data2.csv'
TEST_DATA_FILE = 'test_sensor_data2.csv'
SAMPLING_RATE = 50
WINDOW_SIZE = 50
OVERLAP = 25
USE_2D_CNN = True

def load_and_preprocess_data(filepath, is_train_data=True):
    try:
        df = pd.read_csv(filepath)
        print(f"\nデータ読み込み成功: {filepath}")
        print(df.head())
        print(df.info())

        if 'Label' in df.columns and df['Label'].dtype == 'object':
            df['Label'] = df['Label'].apply(lambda x: 1 if x == '動作中' else 0)
        elif 'label' in df.columns and df['label'].dtype == 'object':
            df['label'] = df['label'].apply(lambda x: 1 if x == '動作中' else 0)

    except FileNotFoundError:
        print(f"{filepath} が見つかりません。デモンストレーション用のダミーデータを作成します。")
        num_samples = 10000
        time = np.arange(num_samples) / SAMPLING_RATE

        acc_x_walk = np.sin(2 * np.pi * 1.5 * time) * 0.5 + np.random.normal(0, 0.1, num_samples)
        acc_y_walk = np.cos(2 * np.pi * 1.5 * time) * 0.5 + np.random.normal(0, 0.1, num_samples)
        acc_z_walk = np.sin(2 * np.pi * 3 * time) * 0.3 + 9.8 + np.random.normal(0, 0.1, num_samples)
        gyro_x_walk = np.cos(2 * np.pi * 1.5 * time) * 10 + np.random.normal(0, 5, num_samples)
        gyro_y_walk = np.sin(2 * np.pi * 1.5 * time) * 10 + np.random.normal(0, 5, num_samples)
        gyro_z_walk = np.cos(2 * np.pi * 3 * time) * 5 + np.random.normal(0, 2, num_samples)

        acc_x_still = np.random.normal(0, 0.05, num_samples)
        acc_y_still = np.random.normal(0, 0.05, num_samples)
        acc_z_still = np.random.normal(0, 0.05, num_samples) + 9.8
        gyro_x_still = np.random.normal(0, 0.1, num_samples)
        gyro_y_still = np.random.normal(0, 0.1, num_samples)
        gyro_z_still = np.random.normal(0, 0.1, num_samples)

        df_walk_segment = pd.DataFrame({
            'Acc_X': acc_x_walk, 'Acc_Y': acc_y_walk, 'Acc_Z': acc_z_walk,
            'Gyr_X': gyro_x_walk, 'Gyr_Y': gyro_y_walk, 'Gyr_Z': gyro_z_walk,
            'Label': 1
        })
        df_still_segment = pd.DataFrame({
            'Acc_X': acc_x_still, 'Acc_Y': acc_y_still, 'Acc_Z': acc_z_still,
            'Gyr_X': gyro_x_still, 'Gyr_Y': gyro_y_still, 'Gyr_Z': gyro_z_still,
            'Label': 0
        })

        segment_length = num_samples // 4
        df_segments = [
            df_walk_segment.iloc[:segment_length],
            df_still_segment.iloc[segment_length:segment_length*2],
            df_walk_segment.iloc[segment_length*2:segment_length*3],
            df_still_segment.iloc[segment_length*3:]
        ]
        df = pd.concat(df_segments, ignore_index=True)

        print("ダミーデータ作成完了:")
        print(df.head())
        print(df.tail())
        print(df['Label'].value_counts())

    features = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
    if 'Label' in df.columns:
        labels = df['Label'].values
    else:
        labels = df['label'].values

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    X_data = []
    y_data = []

    for i in range(0, len(df) - WINDOW_SIZE + 1, OVERLAP):
        window_data = df[features].iloc[i:i + WINDOW_SIZE].values
        window_labels = labels[i:i + WINDOW_SIZE]

        if np.all(window_labels == window_labels[0]):
            X_data.append(window_data)
            y_data.append(window_labels[0])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"\n作成されたウィンドウ数: {len(X_data)}")
    print(f"各ウィンドウの形状: {X_data.shape[1:]}")

    return X_data, y_data

print("--- 学習データの読み込みと前処理 ---")
X_train_raw, y_train = load_and_preprocess_data(TRAIN_DATA_FILE, is_train_data=True)

print("\n--- テストデータの読み込みと前処理 ---")
X_test_raw, y_test = load_and_preprocess_data(TEST_DATA_FILE, is_train_data=False)

if USE_2D_CNN:
    X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], X_train_raw.shape[2], 1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], X_test_raw.shape[2], 1)
    print(f"2D CNN入力のためのX_trainの形状: {X_train.shape}")
    print(f"2D CNN入力のためのX_testの形状: {X_test.shape}")
else:
    X_train = X_train_raw
    X_test = X_test_raw
    print(f"1D CNN入力のためのX_trainの形状: {X_train.shape}")
    print(f"1D CNN入力のためのX_testの形状: {X_test.shape}")

print(f"\n最終的な訓練データの形状: {X_train.shape}, ラベル数: {y_train.shape}")
print(f"最終的なテストデータの形状: {X_test.shape}, ラベル数: {y_test.shape}")

def build_cnn_lstm_model(input_shape, num_classes, use_2d_cnn=False):
    model = models.Sequential()

    if use_2d_cnn:
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 1)))
        model.add(layers.Dropout(0.3))

        _, conv_output_timesteps, conv_output_features, conv_output_channels = model.output_shape
        model.add(layers.Reshape((conv_output_timesteps, conv_output_features * conv_output_channels)))

    else:
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.3))

    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model

input_shape = X_train.shape[1:]
num_classes = 1

model = build_cnn_lstm_model(input_shape, num_classes, use_2d_cnn=USE_2D_CNN)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

def add_gaussian_noise(data, scale=0.02):
    noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
    return data + noise

X_train_augmented = X_train.copy()
y_train_augmented = y_train.copy()

X_train_augmented = np.concatenate((X_train_augmented, add_gaussian_noise(X_train, scale=0.05)), axis=0)
y_train_augmented = np.concatenate((y_train_augmented, y_train), axis=0)

print(f"\nデータ拡張後の訓練データの形状: {X_train_augmented.shape}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

print("\nモデル学習を開始します...")
history = model.fit(X_train_augmented, y_train_augmented,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=1)

print("\nモデル学習が完了しました。")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\nテストデータでのモデル評価を開始します...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"テストデータでの損失: {loss:.4f}")
print(f"テストデータでの精度: {accuracy:.4f}")

predictions = (model.predict(X_test) > 0.5).astype(int)
print("\n分類レポート:")
print(classification_report(y_test, predictions, target_names=['静止中', '動作中']))
print("\n混同行列:")
print(confusion_matrix(y_test, predictions))

model.save('motion_still_discrimination_model_v2.h5')
print("\nモデルが 'motion_still_discrimination_model_v2.h5' として保存されました。")
