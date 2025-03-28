import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import cv2
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. 数据加载与预处理
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 2. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 构建支持向量机模型
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# 4. 预测与评估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)

# # 绘制决策树
# plt.figure(figsize=(20, 10))
# tree.plot_tree(dtc, filled=True, feature_names=[f"像素_{i}" for i in range(64)], 
#               class_names=[str(i) for i in range(10)])
# plt.title('决策树结构')
# plt.show()

# 7. 深度学习CNN模型
# 数据预处理
X_cnn = digits.images.reshape(-1, 8, 8, 1)
y_cnn = to_categorical(y)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.3, random_state=42)

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32, verbose=0)

# 8. 手写体识别
# 加载手写体图像
img = cv2.imread('/Users/caleb/归档/Trae/001/实验/手写体.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)  # 提高分辨率
# img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊降噪
# img = cv2.bitwise_not(img)  # 反转颜色
# img = img.astype('float32') / 255.0  # 归一化

# 为SVM准备数据
img_svm = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
img_svm = img_svm * 16  # 缩放到0-16范围
img_svm_array = img_svm.reshape(1, -1)

# 为CNN准备数据
img_cnn = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
img_cnn = img_cnn.reshape(1, 8, 8, 1)

# 预测手写数字
svm_prediction = clf.predict(img_svm_array)
cnn_prediction = np.argmax(model.predict(img_cnn))
print(f"SVM识别结果为:{svm_prediction}")
print(f"CNN识别结果为: {cnn_prediction}")