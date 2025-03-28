from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 加载数据集
data = load_iris()
X, y = data.data, data.target
# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 创建决策树模型
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
# 预测评估
pred = clf.predict(X_test)
print(f'模型准确率: {accuracy_score(y_test, pred):.2%}')
# 可视化决策树
plt.figure(figsize=(15,10))
plot_tree(clf, 
          feature_names=data.feature_names, 
          class_names=data.target_names,
          filled=True,
          rounded=True)
plt.savefig('decision_tree.png')
plt.show()
