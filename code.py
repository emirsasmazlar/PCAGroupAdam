# Import necessary libraries
import tensorflow as tf # type: ignore
import shap
import numpy as np
import pandas as pd
import GEOparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Custom Optimizer: PCAGroupAdam
class PCAGroupAdam(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=0.001, group_weights=None, name="PCAGroupAdam", **kwargs):
     super().__init__(learning_rate=learning_rate, name=name, **kwargs)
     self.group_weights = group_weights

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
      if self.group_weights is not None:
        group_weights_sum = tf.reduce_sum(self.group_weights)
        normalized_weights = [w / group_weights_sum for w in self.group_weights]
        print(f"Using normalized group weights: {normalized_weights}")

        scaled_grads_and_vars = []
        for i, (grad, var) in enumerate(grads_and_vars):
          if grad is not None:
            weight_index = min(i, len(normalized_weights) - 1)
            scaled_grad = grad * tf.cast(normalized_weights[min(i, len(normalized_weights) - 1)], tf.float32)
            scaled_grads_and_vars.append((scaled_grad, var))
          else:
            scaled_grads_and_vars.append((grad, var))
        return super().apply_gradients(scaled_grads_and_vars)
      return super().apply_gradients(grads_and_vars)
gse = GEOparse.get_GEO("GSE131051", destdir="./")
import pandas as pd

print({gsm.metadata['title'][0]: gsm.table for gsm in gse_5.gsms.values()})

# Create a DataFrame for expression data
expression_data = pd.DataFrame({
    gsm.metadata['title'][0]: gsm.table['VALUE']
    for gsm in gse.gsms.values()
}).T

gene_names = pd.DataFrame({
    gsm.metadata['title'][0]: gsm.table['ID_REF']
    for gsm in gse.gsms.values()
}).T

# Print a snippet of the data
print(expression_data.head())
labels = []
for gsm_name, gsm in gse.gsms.items():
    if "pdx" in gsm.metadata['source_name_ch1'][0].lower():
        labels.append(0)
    else:
        labels.append(1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
A_normalized = scaler.fit_transform(expression_data.values)
from sklearn.decomposition import PCA

pca = PCA(n_components=25)
A_pca = pca.fit_transform(A_normalized)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6), dpi=300)
plt.scatter(A_pca[:, 0], A_pca[:, 1], c=labels_5, cmap='viridis', alpha=0.7)
plt.colorbar(label='Class (0 = Control, 1 = Breast Cancer)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: 2D Visualization')
plt.show()
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6), dpi=300)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), explained_variance.cumsum(), where='mid', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio')
plt.title('Explained Variance by PCA')
plt.legend()
plt.show()
print("Explained Variance Ratio:", pca_5.explained_variance_ratio_)
A = A_pca
#A = A_normalized
B = np.array(labels)

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.3, random_state=42)


#Construct the Model

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model = tf.keras.Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.08)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.08)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.08)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.02,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = PCAGroupAdam(learning_rate=lr_schedule, group_weights=explained_variance)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

#Train the Model on the Dataset GSE131051
history = model.fit(A_train, B_train, epochs=15, batch_size=16, validation_split=0.2, callbacks=[early_stop])
loss, accuracy = model.evaluate(A_test, B_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
a_pred = (model.predict(A_test) > 0.5).astype("int32")
a_proba = model.predict(A_test)
print("Predictions:", a_pred.flatten())
print("True Labels:", B_test)
print(labels_5)

#Analyze and Visualise the Results
conf_matrix = confusion_matrix(B_test, a_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Class 0", "Class 1"])
disp.plot()
report = classification_report(B_test, a_pred, target_names=["Class 0", "Class 1"])
print(report)
f1 = f1_score(B_test, a_pred)
print("F1 SCORE:", f1)
fpr, tpr, thresholds = roc_curve(B_test, a_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
explainer = shap.Explainer(model, A_pca)
shap_values = explainer(A_pca)
shap.summary_plot(shap_values, A_pca, feature_names=[f'PC{i+1}' for i in range(A_pca.shape[1])])
pca_components = pca.components_
gene_shap_values = np.dot(shap_values.values, pca_components)
mean_gene_shap = np.mean(np.abs(gene_shap_values), axis=0)
gene_names_list = gene_names.iloc[0].tolist()
print(len(gene_names_list), len(mean_gene_shap))
shap_gene_importance = pd.DataFrame({'Gene': gene_names_list, 'SHAP Importance': mean_gene_shap})
shap_gene_importance = shap_gene_importance.sort_values(by='SHAP Importance', ascending=False)
print(shap_gene_importance.head(10))
