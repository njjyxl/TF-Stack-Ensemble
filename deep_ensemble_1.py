import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class BaseModelOptimizer:
    """用于优化和评估基础模型的类"""
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.best_models = {}
        self.param_grids = self._define_param_grids()
    
    def _define_param_grids(self):
        """定义每个基础模型的参数网格"""
        return {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gbt': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            },
            'xgb': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def optimize_base_models(self, X, y):
        """优化所有基础模型"""
        # 确保X和y都是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        base_models = {
            'rf': RandomForestClassifier(random_state=42),
            'gbt': GradientBoostingClassifier(random_state=42),
            'xgb': XGBClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        for name, model in base_models.items():
            print(f"\n优化 {name} 模型...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[name],
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X, y)
            
            self.best_models[name] = grid_search.best_estimator_
            print(f"{name} 最佳参数:", grid_search.best_params_)
            print(f"{name} 最佳交叉验证分数:", grid_search.best_score_)
            
            # 计算并打印详细的交叉验证指标
            self._evaluate_model(name, self.best_models[name], X, y)
    
    def _evaluate_model(self, name, model, X, y):
        """使用交叉验证评估单个模型的性能"""
        # 确保X和y都是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            metrics['accuracy'].append(accuracy_score(y_val_fold, y_pred))
            metrics['precision'].append(precision_score(y_val_fold, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_val_fold, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_val_fold, y_pred, average='weighted'))
        
        print(f"\n{name} 交叉验证结果:")
        for metric, scores in metrics.items():
            print(f"{metric}:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std: {np.std(scores):.4f}")
    
    def plot_model_comparison(self, X, y):
        """绘制所有模型的性能比较"""
        # 确保X和y都是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        model_scores = {name: [] for name in self.best_models.keys()}
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.best_models.items():
            scores = []
            precisions = []
            recalls = []
            f1s = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                y_pred = model.predict(X_val)
                
                scores.append(accuracy_score(y_val, y_pred))
                precisions.append(precision_score(y_val, y_pred, average='weighted'))
                recalls.append(recall_score(y_val, y_pred, average='weighted'))
                f1s.append(f1_score(y_val, y_pred, average='weighted'))
            
            model_scores[name] = [
                np.mean(scores),
                np.mean(precisions),
                np.mean(recalls),
                np.mean(f1s)
            ]
        
        # 创建性能比较图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (name, scores) in enumerate(model_scores.items()):
            plt.bar(x + i * width, scores, width, label=name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, metrics)
        plt.legend()
        plt.savefig(f'Model Performance Comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, X, y):
        """为所有模型绘制ROC曲线"""
        # 确保X和y都是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        plt.figure(figsize=(10, 8))
        
        for name, model in self.best_models.items():
            y_scores = np.zeros((len(y), len(np.unique(y))))
            
            # 使用交叉验证获取预测概率
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            for train_idx, val_idx in skf.split(X, y):
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                model.fit(X[train_idx], y[train_idx])
                y_scores[val_idx] = model.predict_proba(X_val)
            
            # 计算每个类的ROC曲线
            n_classes = len(np.unique(y))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y == i).astype(int), y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr,
                    label=f'{name} - Class {i} (AUC = {roc_auc:.2f})'
                )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC Curves for All Models.pdf', dpi=300, bbox_inches='tight')
        plt.close()

class TransformerStackingClassifier:
    def __init__(self, base_models, input_dim=None, num_classes=None):
        self.base_models = base_models
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.meta_learner = None
        self.base_predictions = None
        
    def create_transformer_meta_learner(self, input_dim, num_classes):
        inputs = Input(shape=(input_dim,))
        x = tf.keras.layers.Reshape((1, input_dim))(inputs)
        
        # Multi-head attention blocks
        x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.2)(x)
        
        x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.2)(x)
        
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        # Get predictions from base models using cross-validation
        self.base_predictions = np.column_stack([
            model.predict_proba(X) for _, model in self.base_models
        ])
        
        # Create and train meta-learner
        if self.input_dim is None:
            self.input_dim = self.base_predictions.shape[1]
        if self.num_classes is None:
            self.num_classes = len(np.unique(y))
            
        self.meta_learner = self.create_transformer_meta_learner(
            self.input_dim, 
            self.num_classes
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        self.meta_learner.fit(
            self.base_predictions,
            y,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
    
    def predict(self, X):
        # Get predictions from base models
        meta_features = np.column_stack([
            model.predict_proba(X) for _, model in self.base_models
        ])
        
        # Use meta-learner to make final predictions
        return np.argmax(self.meta_learner.predict(meta_features), axis=1)

def main():
    # 加载和预处理数据
    df = pd.read_csv('data/expr_matrix.csv')
    df_1 = df.drop('Unnamed: 0', axis=1)
    
    le = LabelEncoder()
    df_1['Cancer_tumor'] = le.fit_transform(df_1['Cancer_tumor'])
    
    X = df_1.drop('Cancer_tumor', axis=1)
    y = df_1['Cancer_tumor']
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 优化和评估基础模型
    base_optimizer = BaseModelOptimizer()
    base_optimizer.optimize_base_models(X_train_scaled, y_train)
    
    # 绘制模型比较图
    print("\n绘制模型性能比较...")
    base_optimizer.plot_model_comparison(X_train_scaled, y_train)
    
    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    base_optimizer.plot_roc_curves(X_train_scaled, y_train)
    
    # 使用优化后的基础模型创建堆叠分类器
    optimized_base_models = [
        (name, model) for name, model in base_optimizer.best_models.items()
    ]
    
    stacking_clf = TransformerStackingClassifier(
        base_models=optimized_base_models,
        num_classes=len(np.unique(y))
    )
    
    # 训练堆叠分类器
    print("\n训练堆叠分类器...")
    stacking_clf.fit(X_train_scaled, y_train)
    
    # 评估最终模型
    y_pred = stacking_clf.predict(X_test_scaled)
    print("\n最终模型评估结果:")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred, average='weighted'):.4f}")

if __name__ == "__main__":
    main()