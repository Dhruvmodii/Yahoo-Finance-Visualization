# Yahoo-Finance-Visualization
This is a Streamlit-based finance visualization and forecasting tool that predicts the next **30 days of stock closing prices** using a pre-trained deep learning model (`keras model.h5`). It also provides useful financial insights like moving averages, daily return %, and 30-day rolling volatility.

Live Demo
 https://yahoofinance-v.streamlit.app/
 
🚀 Features
- 📊 Historical stock closing prices with 365, 730, and 1095-day Moving Averages  
- 🔄 Daily price change percentage chart  
- 📉 30-day rolling volatility plot  
- 🔮 Predict the next 30 days of closing prices using a deep learning model (LSTM)  
- 📐 Dynamic line charts with future predictions  
- 🧮 Uses Yahoo Finance API to fetch data  

🗃️ File Structure

 📁stock-price-predictor/
 
   ├── app.py 
     
   ├── keras model.h5 
   
   ├── requirements.txt 

💻 Setup Instructions
1. Clone the Repository
    git clone https://github.com/your-username/stock-price-predictor.git
    cd stock-price-predictor
2. Create & Activate Virtual Environment (optional but recommended)
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
3. Install Required Packages
    pip install -r requirements.txt
4. Run the App
    streamlit run app.py

🧠 Model
The keras model.h5 is a pre-trained LSTM model trained on stock price sequences. It uses the past 100 days of closing prices to predict the next 30 days.

📦 Requirements
See requirements.txt for all dependencies. Major ones include:
streamlit
yfinance
tensorflow
pandas
scikit-learn
matplotlib
numpy

📸 Preview
<img width="1919" height="651" alt="image" src="https://github.com/user-attachments/assets/f4ba6f2e-0450-4165-9cc0-06a10329776d" />
<img width="831" height="502" alt="image" src="https://github.com/user-attachments/assets/ba833ea4-537e-4eeb-b853-ba673696f531" />
<img width="666" height="653" alt="image" src="https://github.com/user-attachments/assets/1ff86870-5604-46d7-9bc4-10d9bb16d24d" />
<img width="673" height="649" alt="image" src="https://github.com/user-attachments/assets/9d2e02c7-9e2c-41f8-9102-44e8a46e165f" />


🧑‍💻 Author
Dhruv Modi

GitHub: @Dhruvmodii

⚠️ Disclaimer
This app is for educational purposes only. It does not offer financial advice. Always consult a professional before making trading decisions.
