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
    def __init__(self, learning_rate=0.001, group_weights=None, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.group_weights = group_weights

    def apply_gradients(self, grads_and_vars, **kwargs):
        if self.group_weights is not None:
            normalized_weights = [
                w / tf.reduce_sum(self.group_weights) for w in self.group_weights
            ]
            scaled_grads_and_vars = [
                (grad * tf.cast(normalized_weights[min(i, len(normalized_weights) - 1)], tf.float32), var)
                if grad is not None else (grad, var)
                for i, (grad, var) in enumerate(grads_and_vars)
            ]
            return super().apply_gradients(scaled_grads_and_vars, **kwargs)
        return super().apply_gradients(grads_and_vars, **kwargs)

# Function to preprocess data
def preprocess_data(gse_id, n_components=25):
    gse = GEOparse.get_GEO(gse_id, destdir="./")
    expression_data = pd.DataFrame({
        gsm.metadata['title'][0]: gsm.table['VALUE'] for gsm in gse.gsms.values()
    }).T
    labels = [
        1 if "tumor" in gsm.metadata['characteristics_ch1'][0].lower() else 0
        for gsm in gse.gsms.values()
    ]
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(expression_data.values)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_normalized)
    explained_variance = pca.explained_variance_ratio_
    return train_test_split(X_pca, labels, test_size=0.3, random_state=42), explained_variance, pca, expression_data

# Load and preprocess dataset
(X_train, X_test, y_train, y_test), explained_variance, pca, expression_data = preprocess_data("GSE15824")

# Build and compile the model
def build_model(input_dim, optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.08)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.08)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.08)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training configuration
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.02, decay_steps=1000, decay_rate=0.96, staircase=True
)
optimizer = PCAGroupAdam(learning_rate=lr_schedule, group_weights=explained_variance)
model = build_model(input_dim=X_train.shape[1], optimizer=optimizer)

# Train the model
history = model.fit(
    X_train, y_train, epochs=15, batch_size=16, validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Evaluate and visualize results
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# SHAP Analysis
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Summary plot for SHAP
shap.summary_plot(shap_values, X_test, feature_names=[f'PC{i+1}' for i in range(X_test.shape[1])])

# Gene-Level SHAP Analysis
pca_components = pca.components_
gene_shap_values = np.dot(shap_values.values, pca_components)
mean_gene_shap = np.mean(np.abs(gene_shap_values), axis=0)
gene_names = expression_data.columns.tolist()

# Create DataFrame of gene SHAP importance
shap_gene_importance = pd.DataFrame({'Gene': gene_names, 'SHAP Importance': mean_gene_shap})
shap_gene_importance = shap_gene_importance.sort_values(by='SHAP Importance', ascending=False)
print("Top Genes by SHAP Importance:")
print(shap_gene_importance.head(10))
