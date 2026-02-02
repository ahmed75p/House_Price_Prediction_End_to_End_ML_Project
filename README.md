# House Price Prediction Project 🏠💰




---

## 📌 Project Overview
This project predicts house prices and explains them using machine learning and a Large Language Model (LLM).
It combines **data analysis, clustering, classification, regression**, and a **Streamlit interactive app** for end-users.  



**Goals:**
- Clean and explore the house price dataset.
- Extract business insights for investment strategy.
- Build ML models to classify and predict house prices.
- Use LLM to explain predictions in human-readable language.
- Deploy an interactive Streamlit app.

---

## 1️⃣ Data Cleaning & Understanding
- Removed outliers and errors to ensure data quality.
- Studied key features: bedrooms, bathrooms, sqft_living, sqft_above, waterfront, view, condition, location.
- Performed **Exploratory Data Analysis (EDA)** using:
  - `matplotlib`  
  - `seaborn`  
  - `plotly`



### 🔍 Key Business Recommendations
- **House Types:** Focus on 3-bedroom and 2.5-bathroom houses (most in-demand).  
- **Living Area:** Larger `sqft_living` and `sqft_above` are the strongest price drivers.  
- **Premium Features:** Waterfront and high view ratings command higher prices.  
- **Condition:** Renovated/well-maintained houses fetch higher prices, regardless of age.  
- **Location Strategy:** 
  - For volume: Seattle  
  - For high-value returns: Clyde Hill & Mercer Island  
- **Basements & Lot Size:** Basements moderately increase price; lot size has minimal effect.

---

## 2️⃣ Clustering
- **Model:** Agglomerative Hierarchical Clustering  
- **Clusters:** 2 clusters identified:
  1. Standard Homes  
  2. Premium Homes  

> Helps segment the market for targeted investment.

---

## 3️⃣ Classification
- **Model:** Support Vector Classifier (SVC)  
- **Accuracy:** 98%  
- Predicts if a house belongs to Standard or Premium cluster.

---

## 4️⃣ Regression
- **Model:** Ridge Regression   
- Predicts house price based on features.

---

## 5️⃣ Large Language Model (LLM)
- **API:** GROQ API (LLaMA-3.3-70b-versatile)  
- Provides human-readable explanations for each predicted price:  
  1. Why was this price predicted? (Based on house features)  
  2. Is it realistic compared to the market? Explain why.

---

## 6️⃣ Streamlit App
Interactive app lets users:  
1. Enter house features  
2. Predict house price  
3. Identify the cluster (Standard / Premium)  
4. Get LLM explanation for the predicted price  

**Live Streamlit App:** [Open Here](https://housepricepredictionendtoendmlproject-h3k9wasehygz2b8rf4fpv8.streamlit.app/)

---
## 7- Dashboard
Interactive dashboard

**image1:** ![click here](https://github.com/ahmed75p/House_Price_Prediction_End_to_End_ML_Project/blob/main/images/Dashboard1.png)

**image2:** [click here](https://github.com/ahmed75p/House_Price_Prediction_End_to_End_ML_Project/blob/main/images/Dashboard2.png)

---

## 🗂️ Project Structure

House-Price-Prediction/
│

├─ code/ # Notebooks for data analysis and modeling

├─ data/ # Datasets

├─ models/ # Saved ML models

├─ images/ # Project visuals and screenshots

├─ presentation/ # Real Estate

├─ streamlit_app/ # Streamlit app code

├─ Dashboard/ # real estate

├─ requirements.txt # Python dependencies

└─ README.md # Project overview


---

## ⚙️ How to Run Locally
```
1. Clone the repository:
git clone https://github.com/username/House-Price-Prediction.git


2. Install dependencies:
pip install -r requirements.txt


3. Run Streamlit app:
streamlit run streamlit_app/app.py


4.Add your API key (locally):
GROQ_API_KEY = "your-api-key-here"
```
---
Summary

This project demonstrates end-to-end data analysis, machine learning, and AI explainability for house price prediction,
providing actionable business insights and an interactive user experience.

---

### 🧑‍💻 Author
**Ahmed Mostafa**  
Data Analyst 
[LinkedIn Profile](https://www.linkedin.com/in/ahmed-mostafa-841412250/)  

📧 Email:
👉 ahmedmostafa75p@gmail.com
