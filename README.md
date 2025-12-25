Solar Irradiation Forecasting Using Genetic Algorithms
This document describes the reproducible implementation of the solar irradiation forecasting framework using Linear Regression (LR), Extreme Gradient Boosting (XGB), and Genetic Algorithm (GA)–optimized XGB models.
Overview
The repository contains a fully reproducible Python implementation aligned with the manuscript titled:
“Solar Irradiation Forecasting Using Genetic Algorithms: GA-Optimized Machine Learning Models for Short-Term Solar Irradiation Prediction.”
Key Features
•	Strict temporal split (Training: 2018–2019, Validation: 2020)
•	Random Forest–based feature selection
•	GA-optimized XGBoost (GA-10)
•	Reproduction of all figures and tables from the manuscript
•	Reviewer-safe, deterministic execution
How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Execute the script:
   python solar_ga_forecasting.py
License
MIT License
Citation
Gunasekaran, V., et al. (2025). Solar Irradiation Forecasting Using Genetic Algorithms. F1000Research.
