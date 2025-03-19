import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Định nghĩa đường dẫn dữ liệu
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['null', 'Xin Chao', 'Cam On', 'Xin Loi', 'Tam Biet'])

# Thông số dữ liệu
no_sequences = 30
sequence_length = 30

# Ánh xạ nhãn
label_map = {label: num for num, label in enumerate(actions)}

# Load dữ liệu test
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Tách dữ liệu train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Load mô hình đã huấn luyện
model = load_model("action.h5")

# Kiểm tra trên tập test
correct = 0
for i in range(len(x_test)):
    res = model.predict(np.expand_dims(x_test[i], axis=0))  # Dự đoán từng mẫu
    predicted_action = actions[np.argmax(res)]
    actual_action = actions[np.argmax(y_test[i])]
    print(f"🔮 Dự đoán: {predicted_action} | ✅ Thực tế: {actual_action}")

    if predicted_action == actual_action:
        correct += 1

# Tính độ chính xác
accuracy = correct / len(x_test) * 100
print(f"🎯 Độ chính xác trên tập test: {accuracy:.2f}%")
