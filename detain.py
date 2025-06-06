import os
import serial
import struct
import binascii
import ctypes
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# --- グローバル変数と設定 ---
COM_PORT = "COM6"
BAUD_RATE = 115200

MODEL_PATH = 'motion_still_discrimination_model_v2.h5'
SCALER_PATH = 'scaler_v2.joblib'

ACC_DISPLAY_BUFFER_SIZE = 100

# --- ユーティリティ関数 ---
def twos_complement(value, bits):
    # 符号拡張
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value

def read_data(ser):
    # シリアルポートからデータを読み込む
    while running:
        byte = ser.read(1)
        if not byte:
            continue
        if ord(byte) == 0x9A:
            break

    if ord(ser.read(1)) == 0x80:
        try:
            # 各センサーバイトを読み込み
            ticktime_bytes = ser.read(4)
            acc_x_bytes = ser.read(3)
            acc_y_bytes = ser.read(3)
            acc_z_bytes = ser.read(3)
            gyr_x_bytes = ser.read(3)
            gyr_y_bytes = ser.read(3)
            gyr_z_bytes = ser.read(3)

            # データ長を確認
            if all(len(b) == (4 if i==0 else 3) for i, b in enumerate([ticktime_bytes, acc_x_bytes, acc_y_bytes, acc_z_bytes, gyr_x_bytes, gyr_y_bytes, gyr_z_bytes])):
                # バイトデータを数値に変換
                ticktime = struct.unpack('<I', ticktime_bytes)[0]
                acc_x = twos_complement(int.from_bytes(acc_x_bytes, byteorder='little', signed=False), 24)
                acc_y = twos_complement(int.from_bytes(acc_y_bytes, byteorder='little', signed=False), 24)
                acc_z = twos_complement(int.from_bytes(acc_z_bytes, byteorder='little', signed=False), 24)
                gyr_x = twos_complement(int.from_bytes(gyr_x_bytes, byteorder='little', signed=False), 24)
                gyr_y = twos_complement(int.from_bytes(gyr_y_bytes, byteorder='little', signed=False), 24)
                gyr_z = twos_complement(int.from_bytes(gyr_z_bytes, byteorder='little', signed=False), 24)

                return ticktime, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
        except struct.error as e:
            print(f"データパックの解析エラー: {e}")
        except Exception as e:
            print(f"データ読み込み中の予期せぬエラー: {e}")
    return None

def data_thread(ser):
    # バックグラウンドでシリアルデータを読み込み続ける
    global latest_sensor_data
    while running:
        current_data = read_data(ser)
        if current_data:
            latest_sensor_data = current_data

def on_key(event):
    # Escキーで終了
    global running
    if event.keysym == 'Escape':
        running = False

def on_closing():
    # ウィンドウクローズ時に終了
    global running
    running = False
    root.destroy()

def calculate_frequency(data_list, sample_interval):
    # ゼロクロス周波数を計算
    if not data_list or len(data_list) < 2:
        return 0.0

    zero_crossings = np.where(np.diff(np.sign(data_list)))[0]

    if len(zero_crossings) > 1:
        total_samples = zero_crossings[-1] - zero_crossings[0]
        num_periods = len(zero_crossings) - 1 

        if total_samples > 0 and num_periods > 0:
            samples_per_period = total_samples / num_periods
            period_sec = samples_per_period * sample_interval
            
            frequency = 1.0 / period_sec
            return frequency
    return 0.0

def animate(i):
    # Matplotlibの3Dグラフを更新
    global latest_sensor_data, ax, x_acc_data_buffer, y_acc_data_buffer, z_acc_data_buffer

    if latest_sensor_data:
        _, acc_x, acc_y, acc_z, _, _, _ = latest_sensor_data
        
        # バッファにデータを追加
        x_acc_data_buffer.append(acc_x * 0.1)
        y_acc_data_buffer.append(acc_y * 0.1)
        z_acc_data_buffer.append(acc_z * 0.1)
        
        # バッファサイズを管理
        if len(x_acc_data_buffer) > ACC_DISPLAY_BUFFER_SIZE:
            x_acc_data_buffer.pop(0)
            y_acc_data_buffer.pop(0)
            z_acc_data_buffer.pop(0)

        # グラフをクリアしてプロット
        ax.clear()
        ax.plot(z_acc_data_buffer, y_acc_data_buffer, x_acc_data_buffer, color='b', label="Acceleration Path")
        ax.scatter(z_acc_data_buffer[-1], y_acc_data_buffer[-1], x_acc_data_buffer[-1], color='r', s=50)
        
        ax.legend()
        ax.set_xlabel('Acceleration Z (mg)')
        ax.set_ylabel('Acceleration Y (mg)')
        ax.set_zlabel('Acceleration X (mg)')
        ax.set_title('Real-time 3D Acceleration Data')

        # 周波数計算とGUI表示
        calculated_frequency = calculate_frequency(x_acc_data_buffer, 0.1) 
        frequency_var.set(f"{calculated_frequency:.2f} Hz")

def update_gui():
    # GUIのラベルと判別結果を定期的に更新
    global latest_sensor_data, model, scaler, prediction_var

    if latest_sensor_data:
        # GUI変数にデータをセット
        ticktime, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z = latest_sensor_data
        ticktime_var.set(str(ticktime))
        acc_x_var.set(str(acc_x))
        acc_y_var.set(str(acc_y))
        acc_z_var.set(str(acc_z))
        gyr_x_var.set(str(gyr_x))
        gyr_y_var.set(str(gyr_y))
        gyr_z_var.set(str(gyr_z))
        
        # リアルタイム判別
        features_for_prediction = np.array([[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]])
        
        try:
            features_scaled = scaler.transform(features_for_prediction)
        except Exception as e:
            print(f"スケーラー変換エラー: {e}. features_for_predictionの形状: {features_for_prediction.shape}")
            prediction_var.set("エラー")
            root.after(100, update_gui)
            return

        input_data_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        
        prediction_prob = model.predict(input_data_reshaped, verbose=0)[0][0]
        prediction = (prediction_prob > 0.5).astype(int)

        # 判別結果を表示
        if prediction == 0:
            prediction_var.set(f"静止中 (確率: {1-prediction_prob:.2f})")
        else:
            prediction_var.set(f"動作中 (確率: {prediction_prob:.2f})")
        
    root.after(100, update_gui)

# --- メイン処理 ---
def main():
    global running, latest_sensor_data, x_acc_data_buffer, y_acc_data_buffer, z_acc_data_buffer, ax, root, model, scaler
    running = True
    latest_sensor_data = None

    x_acc_data_buffer, y_acc_data_buffer, z_acc_data_buffer = [], [], []

    # シリアルポート設定とオープン
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1.0)
        if not ser.is_open:
            ser.open() 
        print(f"シリアルポート {COM_PORT} をオープンしました。")
    except serial.SerialException as e:
        print(f"エラー: シリアルポート {COM_PORT} を開けませんでした。{e}")
        print("正しいCOMポートが設定されているか、デバイスが接続されているか確認してください。")
        return

    # モデルとスケーラーの読み込み
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"モデル '{MODEL_PATH}' とスケーラー '{SCALER_PATH}' を読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: モデルまたはスケーラーファイルが見つかりません。{e}")
        print("モデル学習プログラムを先に実行し、ファイルを生成してください。")
        ser.close()
        return
    except Exception as e:
        print(f"エラー: モデルまたはスケーラーの読み込み中に問題が発生しました。{e}")
        ser.close()
        return

    # Tkinter GUIのセットアップ
    root = tk.Tk()
    root.title("リアルタイムセンサーデータ & 動作判別")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # GUI変数
    global ticktime_var, acc_x_var, acc_y_var, acc_z_var, gyr_x_var, gyr_y_var, gyr_z_var, frequency_var, prediction_var
    ticktime_var = tk.StringVar()
    acc_x_var = tk.StringVar()
    acc_y_var = tk.StringVar()
    acc_z_var = tk.StringVar()
    gyr_x_var = tk.StringVar()
    gyr_y_var = tk.StringVar()
    gyr_z_var = tk.StringVar()
    frequency_var = tk.StringVar()
    prediction_var = tk.StringVar(value="待機中...")

    # GUIフレームとラベルの配置
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)

    ttk.Label(frame, text="TickTime:").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=ticktime_var).grid(row=0, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Acceleration X:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=acc_x_var).grid(row=1, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Acceleration Y:").grid(row=2, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=acc_y_var).grid(row=2, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Acceleration Z:").grid(row=3, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=acc_z_var).grid(row=3, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Gyro X:").grid(row=4, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=gyr_x_var).grid(row=4, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Gyro Y:").grid(row=5, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=gyr_y_var).grid(row=5, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Gyro Z:").grid(row=6, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=gyr_z_var).grid(row=6, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="Zero-Crossing Frequency (Acc_X):").grid(row=7, column=0, sticky=tk.W, pady=2)
    ttk.Label(frame, textvariable=frequency_var).grid(row=7, column=1, sticky=tk.E, pady=2)
    ttk.Label(frame, text="**動作判別結果:**").grid(row=8, column=0, sticky=tk.W, pady=5)
    ttk.Label(frame, textvariable=prediction_var, font=('Arial', 12, 'bold'), foreground='blue').grid(row=8, column=1, sticky=tk.E, pady=5)

    # データ読み込みスレッドの開始
    data_thread_instance = threading.Thread(target=data_thread, args=(ser,), daemon=True)
    data_thread_instance.start()
    
    # キーボードイベントバインド
    root.bind_all('<Key>', on_key)
    
    # Matplotlib 3Dグラフのセットアップ
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, animate, interval=100)
    plt.show(block=False)
    
    # GUIの定期更新を開始
    root.after(100, update_gui)
    print("データ受信とリアルタイム判別を開始しました。")
    print("GUIウィンドウを閉じるか、Escキーを押すと終了します。")
    
    # Tkinterメインループの開始
    root.mainloop()
    
    # 終了処理
    running = False
    data_thread_instance.join()
    ser.close()
    plt.close(fig)
    print("アプリケーションを終了しました。")

if __name__ == '__main__':
    main()
