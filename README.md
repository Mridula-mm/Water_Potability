# 💧 Smart Water Quality Monitoring System

An AI-powered, deploy-ready application designed to predict water potability using multiple Machine Learning architectures. This project bridges the gap between environmental data science and public health safety.

## 🌍 Project Overview
Ensuring access to clean water is a cornerstone of public health. This system analyzes nine chemical properties of water to distinguish between **Potable** (Safe) and **Non-Potable** (Unsafe) samples. By utilizing high-accuracy classification models, it provides a reliable tool for smart city monitoring and IoT-based water testing.

## 🚀 Key Features
- **Multi-Model Comparison:** Automatically trains and evaluates KNN, Decision Tree, Random Forest, AdaBoost, SVM, and XGBoost.
- **Automated Best Model Selection:** Identifies and deploys the highest-performing model dynamically.
- **Data Balancing (SMOTE):** Handles class imbalance to ensure fair and accurate predictions.
- **Interactive Dashboard:** Built with **Streamlit** and **Plotly** for real-time data visualization.
- **Explainable AI:** Feature Importance insights to show which chemical properties (like pH or Chloramines) drive the prediction.

## 🧠 Technical Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn
- **Visualization:** Plotly, Streamlit
- **Environment:** VS Code / Jupyter Notebook

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/water-potability-ai.git](https://github.com/yourusername/water-potability-ai.git)
   cd water-potability-ai
