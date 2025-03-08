# Automated Financial Forecasting & Anomaly Detection from SEC Filings

## Project Overview
This project implements an end-to-end pipeline for forecasting a key financial metric, **AccountsPayableCurrent**, using real-world SEC EDGAR data. The pipeline includes:

- **Data Engineering & Preprocessing:**  
  Extraction, cleaning, merging, and daily aggregation of a large SEC dataset (~13GB) into a continuous time series.

- **Clustering & Anomaly Detection:**  
  Standardization and application of K-Means clustering on daily data to group similar financial behaviors. Anomalies are detected based on the distance from cluster centers, with rule-based explanations comparing individual values to cluster averages.

- **Forecasting Models:**  
  - **Prophet:**  
    A baseline model capturing overall trend and seasonality (MAPE ≈ 11.34%).
  - **Hybrid Model (Prophet + XGBoost):**  
    Corrects residual errors from Prophet using XGBoost, dramatically reducing forecasting errors (MAPE ≈ 0.67%).
  - **Deep Learning Models:**  
    - **LSTM:**  
      A recurrent neural network model that captures temporal dependencies in the data.
    - **Transformer:**  
      An advanced model leveraging self-attention mechanisms, achieving a MAPE of ≈ 1.23% on the original scale.

- **Visualization & Interactivity:**  
  Interactive dashboards built with ipywidgets allow users to explore data by selecting specific years and dates, displaying clustering results, anomaly flags, and forecast details.

## Dataset
The project uses the **SEC EDGAR Company Facts** dataset (September 2023) from Kaggle:

[SEC EDGAR Company Facts - Kaggle](https://www.kaggle.com/datasets/jamesglang/sec-edgar-company-facts-september2023)

**Instructions:**
1. Download the dataset from the above link.
2. Place the dataset in a folder named `data/` in the repository.
3. Update the file paths in the notebook accordingly.

## Repository Contents
- **Financial_Forecasting.ipynb**  
  A comprehensive Jupyter Notebook that includes:
  - Data extraction, cleaning, and daily aggregation of SEC EDGAR data.
  - K-Means clustering for anomaly detection and rule-based explanation of anomalies.
  - Forecasting with Prophet, a hybrid Prophet+XGBoost model, and deep learning models (LSTM and Transformer).
  - Evaluation of models using metrics such as MSE, RMSE, MAE, and MAPE.
  - Interactive dashboards and visualizations for exploring results.

## Requirements
Install the required Python libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn prophet xgboost tensorflow torch ipywidgets
```

## How to Run
1. **Download the Dataset:**
   - Download the SEC EDGAR Company Facts dataset from Kaggle and place it in the `data/` folder.
2. **Open the Notebook:**
   - Open `Financial_Forecasting.ipynb` in Jupyter Notebook or JupyterLab.
3. **Execute the Notebook:**
   - Run the cells sequentially to perform data preprocessing, clustering & anomaly detection, forecasting with Prophet, Hybrid, LSTM, and Transformer models, and evaluation.

## Key Results
- **Prophet Model:** MAPE ≈ 11.34%
- **Hybrid Model (Prophet + XGBoost):** MAPE ≈ 0.67%
- **Transformer Model:** MAPE ≈ 1.23%
- **LSTM Model:** Competitive performance (detailed metrics provided in the notebook)

## Insights & Impact
- **Data Engineering:**
  - Successfully managed and preprocessed a large-scale SEC dataset (~13GB) to create a robust daily time series.

- **Anomaly Detection:**
  - K-Means clustering effectively segmented the financial data, and a rule-based approach was used to flag and explain anomalies based on cluster distances.

- **Forecasting Accuracy:**
  - The hybrid model significantly improved forecasting accuracy by correcting Prophet's residuals, while deep learning models (LSTM and Transformer) captured complex temporal dependencies.

- **Practical Applications:**
  - This pipeline provides actionable insights for financial risk management and strategic decision-making, demonstrating advanced skills in data science, machine learning, and deep learning.

## Contact
For questions or further discussion, please contact [Anubhab Bhattacharya] at [anubhabb.cse.ug@jadavpuruniversity.in].
