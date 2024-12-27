# PCAGroupAdam: A Custom Optimizer for Dimensionality-Reduced Gene Expression Data
PCAGroupAdam is a custom optimizer designed to enhance deep learning performance on high-dimensional datasets by leveraging Principal Component Analysis (PCA). This repository implements PCAGroupAdam to optimize models for gene expression data, enabling efficient dimensionality reduction, adaptive gradient scaling, and biological interpretability through SHAP analysis and STRING network insights.
# Clone the repository
git clone https://github.com/<emirsasmazlar>/PCAGroupAdam.git
cd PCAGroupAdam

# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

# Install required libraries
pip install -r requirements.txt

# Preprocess the data, choose a dataset from GEO database
python preprocess_data.py

# Import the optimizer and model
from PCAGroupAdam import PCAGroupAdam, build_model
# Load data and train the model
# Perform SHAP analysis
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

We welcome contributions! Please fork the repository, make changes, and submit a pull request. For major changes, open an issue first to discuss your ideas.
Created by Ahmet Emir Şaşmazlar - emirsasmazlar@gmail.com
GitHub: https://github.com/<emirsasmazlar>

