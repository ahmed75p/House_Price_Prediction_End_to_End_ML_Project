import streamlit as st
import numpy as np
import requests
import joblib
import os

# Load models, encoders, and scalers
models_path = "Models"
model_ridge = joblib.load(os.path.join(models_path, "model_ridge.pkl"))
model_SVC = joblib.load(os.path.join(models_path, "model_SVC.pkl"))

scaler_classification = joblib.load(os.path.join(models_path, "scaler_classification.pkl"))
scaler_regression = joblib.load(os.path.join(models_path, "scaler_regression.pkl"))


encoder_city_class = joblib.load(os.path.join(models_path, "encoder_city_class.pkl"))
encoder_statezip_regression = joblib.load(os.path.join(models_path, "encoder_statezip_regression.pkl"))

# Streamlit UI
st.title("🏠 Intelligent House Price Prediction System")
st.markdown("""
This system predicts house prices using machine learning models trained on real housing data.
Enter the property details below to get an estimated price and market analysis.
""")


# Input fields
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0.5, step=0.5)
sqft_living = st.number_input("Sqft Living", min_value=0 , max_value=4000)
sqft_lot = st.number_input("Sqft Lot", min_value=0 ,max_value=19000)
sqft_above = st.number_input("Sqft Above", min_value=0,max_value=4000)
sqft_basement = st.number_input("Sqft Basement", min_value=0,max_value=1500)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2026, step=1)
yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2026, step=1)
house_age = st.number_input("House Age", min_value=0, step=1)
floors = st.number_input("Floors", min_value=1.0, step=0.5)

waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0,1,2,3,4])
condition = st.selectbox("Condition", [1,2,3,4,5])
month_sold = st.selectbox("Month Sold", list(range(1,13)))
city = st.selectbox("City", encoder_city_class.classes_)
statezip = st.selectbox("State Zip", encoder_statezip_regression.categories_[0])




#chat history resize
MAX_MESSAGES = 200
def trim_history(chat_history):
    system_msg = chat_history[0]
    rest = chat_history[1:]
    if len(rest) > MAX_MESSAGES:
        rest = rest[-MAX_MESSAGES:]

    return [system_msg] + rest

# Predict button
if st.button("Predict"):
    #encoding
    city_encoded = encoder_city_class.transform([city])[0]
    statezip_encoded = encoder_statezip_regression.transform([[statezip]])


    input_half_data_regression = np.array([[
     bedrooms, bathrooms, sqft_living, sqft_lot, floors,waterfront, view,
     condition, sqft_above, sqft_basement, month_sold, house_age
     ]])
    final_input_regression = np.hstack((input_half_data_regression, statezip_encoded))

    #scaling
    final_input_regression_scaled = scaler_regression.transform(final_input_regression)

    #modeling
    price_predicted = model_ridge.predict(final_input_regression_scaled)


    input_half_data_classification = np.array([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, sqft_above, sqft_basement,
    yr_built, yr_renovated, city_encoded , month_sold, house_age
     ]])
    
    final_input_classification = np.hstack((price_predicted.reshape(-1, 1) ,input_half_data_classification ))

    #scaling
    final_input_classification_scaled = scaler_classification.transform(final_input_classification)

    #modeling
    predicted_class = model_SVC.predict(final_input_classification_scaled)
    class_status = "Standard Home" if predicted_class[0] == 0 else "Premium Home"


    #LLm model & API
    house_features = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement,
    'yr_built': yr_built,
    'yr_renovated': yr_renovated,
    'city_encoded': city_encoded,
    'month_sold': month_sold,
    'house_age': house_age
     }
   

    #GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
     }

    prompt = f"""
       You are a real estate expert.

       A machine learning model predicted the house price as ${price_predicted[0]}.

       House details:{house_features} """

    chat_history = [
    {"role": "system", "content":prompt}
     ]


    user_input = """
    Answer both questions exactly:
    question1 : Why was this price predicted ? based on the house features.
    question2 : Is it realistic compared to the market? Explain why?

    Rules:
    - Do NOT repeat the question.
    - Do NOT add any extra text.
    """


    chat_history.append({"role": "user", "content": user_input})
    chat_history = trim_history(chat_history)
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": chat_history,
        "temperature": 0.3
        }
    
    response = requests.post(url, headers=headers, json=data)

# Output fields
    st.metric(label="Predicted Price", value=f"${price_predicted[0]:,.0f}")
    st.metric(label="Predicted class", value=class_status)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        chat_history.append({"role": "assistant", "content": reply})
        st.write("💬 LLM Interpretation:")
        st.write(reply)
    else:
        st.error(f"❌ [Error] {response.status_code}: {response.text}")







