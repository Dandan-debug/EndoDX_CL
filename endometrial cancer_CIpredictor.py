import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 定义文件路径
cl_file_path = '/Users/apple/Desktop/FD/FD/子宫内膜癌EC/实验进展/maldi数据-2/EC-MALDI2/Data analysis/Model/CI模型/920CI.csv'

# 定义保存目录
save_dir = os.path.dirname(cl_file_path)

# 加载CL特征数据
cl_df = pd.read_csv(cl_file_path)
print("CL DataFrame:")
print(cl_df.head())

# 提取CL的目标变量
y = cl_df['target']

# 选择CL的特征列
cl_numeric_features = [
    'CI_age', 'CI_height', 'CI_weight', 'CI_BMI', 'CI_triglycerides',
    'CI_total cholesterol', 'CI_HDL', 'CI_LDL', 'CI_free fatty acids',
    'CI_CA125', 'CI_HE4', 'CI_endometrial thickness'
]
cl_categorical_features = [
    'CI_menopause', 'CI_HRT', 'CI_diabetes', 'CI_hypertension',
    'CI_endometrial heterogeneity', 'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow', 'CI_uterine cavity fluid'
]

# 定义处理流程
cl_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 填补缺失值
    ('scaler', StandardScaler())  # 标准化使数据分布在一定区间
])

cl_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))  # 仅填补缺失值，不进行标准化
])

# 处理CL特征
cl_numeric_processed = cl_numeric_transformer.fit_transform(cl_df[cl_numeric_features])
cl_categorical_processed = cl_categorical_transformer.fit_transform(cl_df[cl_categorical_features])

# 将数值特征和分类特征合并
X_processed = np.hstack((cl_numeric_processed, cl_categorical_processed))

# 获取所有特征名称
feature_names = cl_numeric_features + cl_categorical_features

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3,
                                                    random_state=42, stratify=y)

# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'max_leaves': 127,
    'verbosity': 1,
    'seed': 42,
    'nthread': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'eval_metric': 'logloss'
}

# 初始化XGBoost分类模型
model_xgb = xgb.XGBClassifier(**params_xgb)

# 定义参数网格，用于网格搜索
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.001],
}

# 使用GridSearchCV进行网格搜索和k折交叉验证
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',
    cv=10,
    n_jobs=-1,
    verbose=1
)

# 在训练集上训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 保存训练好的模型
model_save_path = os.path.join(save_dir, 'XGBoostCI.pkl')
joblib.dump(best_model, model_save_path)
print(f"Model saved to {model_save_path}")

# 创建用于保存评估结果的目录
train_cv_dir = os.path.join(save_dir, 'Train_cross_validation')
validation_dir = os.path.join(save_dir, 'Validation_results')
os.makedirs(train_cv_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# === 训练集评估 ===

# 在训练集上直接评估，不使用交叉验证
y_train_pred = best_model.predict(X_train)
y_train_score = best_model.predict_proba(X_train)[:, 1]

# 计算评估指标（训练集）
accuracy_train = accuracy_score(y_train, y_train_pred)
cm_train = confusion_matrix(y_train, y_train_pred)
sensitivity_train = cm_train[1, 1] / (cm_train[1, 1] + cm_train[1, 0])
specificity_train = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1])

# 打印评估指标
print(f"Train Accuracy: {accuracy_train:.2f}")
print(f"Train Sensitivity (Recall): {sensitivity_train:.2f}")
print(f"Train Specificity: {specificity_train:.2f}")

# 保存混淆矩阵图像（训练集）
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=best_model.classes_)
disp_train.plot(cmap=plt.cm.Blues)
plt.title('Train Confusion Matrix')
confusion_matrix_path_train = os.path.join(train_cv_dir, 'Train_Confusion_Matrix.png')
plt.savefig(confusion_matrix_path_train, format='png', dpi=300, bbox_inches='tight')
plt.show()

# 计算ROC曲线（训练集）
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_score)
roc_auc_train = auc(fpr_train, tpr_train)

# 设置图片大小，分辨率和格式（训练集）
plt.figure(figsize=(10, 8), dpi=300)
plt.plot(fpr_train, tpr_train, color='red', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 保存ROC曲线图像（训练集）
roc_curve_path_train = os.path.join(train_cv_dir, 'Train_ROC_Curve.png')
plt.savefig(roc_curve_path_train, format='png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Train ROC curve saved to {roc_curve_path_train}")

# 保存特征重要性表格（训练集）
feature_importances = best_model.feature_importances_
importance_df_train = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df_train.to_csv(os.path.join(train_cv_dir, 'Train_Feature_Importance.csv'), index=False)

# === 验证集评估 ===

# 预测验证集的结果
y_test_pred = best_model.predict(X_test)
y_test_score = best_model.predict_proba(X_test)[:, 1]

# 计算评估指标（验证集）
accuracy_test = accuracy_score(y_test, y_test_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
sensitivity_test = cm_test[1, 1] / (cm_test[1, 1] + cm_test[1, 0])
specificity_test = cm_test[0, 0] / (cm_test[0, 0] + cm_test[0, 1])

# 打印评估指标
print(f"Validation Accuracy: {accuracy_test:.2f}")
print(f"Validation Sensitivity (Recall): {sensitivity_test:.2f}")
print(f"Validation Specificity: {specificity_test:.2f}")

# 保存混淆矩阵图像（验证集）
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=best_model.classes_)
disp_test.plot(cmap=plt.cm.Blues)
plt.title('Validation Confusion Matrix')
confusion_matrix_path_test = os.path.join(validation_dir, 'Validation_Confusion_Matrix.png')
plt.savefig(confusion_matrix_path_test, format='png', dpi=300, bbox_inches='tight')
plt.show()

# 计算ROC曲线（验证集）
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)
roc_auc_test = auc(fpr_test, tpr_test)

# 设置图片大小，分辨率和格式（验证集）
plt.figure(figsize=(10, 8), dpi=300)
plt.plot(fpr_test, tpr_test, color='red', lw=2, label='Validation ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation Receiver Operating Characteristic')
plt.legend(loc="lower right")

# 保存ROC曲线图像（验证集）
roc_curve_path_test = os.path.join(validation_dir, 'Validation_ROC_Curve.png')
plt.savefig(roc_curve_path_test, format='png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Validation ROC curve saved to {roc_curve_path_test}")

# 保存特征重要性表格（验证集）
importance_df_test = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df_test.to_csv(os.path.join(validation_dir, 'Validation_Feature_Importance.csv'), index=False)
