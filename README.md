# Yahoo-Finance-Visualization
This is a Streamlit-based finance visualization. It also provides useful financial insights like moving averages, daily return %, and 30-day rolling volatility.

Live Demo
 https://yahoofinance-v.streamlit.app/
 
🚀 Features
- 📊 Historical stock closing prices with 365, 730, and 1095-day Moving Averages  
- 🔄 Daily price change percentage chart  
- 📉 30-day rolling volatility plot  
- 🧮 Uses Yahoo Finance API to fetch data  

🗃️ File Structure

 📁stock-price-predictor/
 
   ├── app.py 
   
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


📦 Requirements

See requirements.txt for all dependencies. Major ones include:

streamlit

yfinance

pandas

scikit-learn

matplotlib

numpy

📸 Preview
<img width="1919" height="651" alt="image" src="https://github.com/user-attachments/assets/f4ba6f2e-0450-4165-9cc0-06a10329776d" />
<img width="831" height="502" alt="image" src="https://github.com/user-attachments/assets/ba833ea4-537e-4eeb-b853-ba673696f531" />
<img width="666" height="653" alt="image" src="https://github.com/user-attachments/assets/1ff86870-5604-46d7-9bc4-10d9bb16d24d" />


🧑‍💻 Author
Dhruv Modi

GitHub: @Dhruvmodii

⚠️ Disclaimer
This app is for educational purposes only. It does not offer financial advice. Always consult a professional before making trading decisions.
