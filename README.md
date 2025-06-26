# 📈 Stock Price Prediction using LSTM

This project is a web-based application that predicts future stock prices using historical data and a deep learning model (LSTM). It visualizes actual vs predicted stock trends along with technical indicators like EMA (Exponential Moving Averages). Built with Flask and deployed for real-time usage.

---

## 🚀 Features

- 📊 **Stock Price Visualization** with EMAs (20, 50, 100, 200 Days)
- 🤖 **LSTM-based Deep Learning** model for prediction
- 📉 **Prediction vs Original Price** chart with confidence interval
- 📍 **Next Expected Closing Price** display
- 📁 CSV download of full stock data
- 🌐 Deployed on Render (Free Hosting)

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Flask (Python)
- **ML Model:** LSTM (Keras, TensorFlow)
- **Data Source:** [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` API
- **Deployment:** Render / GitHub

---

## 📁 Folder Structure

├── static/ # Plots & downloads (images, CSV)
├── templates/ # HTML template (index.html)
├── app.py # Main Flask application
├── stock_dl_model.h5 # Trained LSTM model
├── best_model.keras # Best checkpoint model
├── requirements.txt # All Python dependencies
├── Procfile # For deployment on Render
└── README.md # You're here
---

## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.8+
- Git
- pip or conda

### 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/Stock_price_prediction.git
cd Stock_price_prediction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

# Run Locally
python app.py

## 🧠 Prediction Example
Input: Any valid stock ticker (e.g., TATAMOTORS.NS, INFY.NS, etc.)

Output:

📈 Historical price chart

🔁 EMAs (trend analysis)

🔮 Next predicted closing price

🎯 Confidence range around prediction

## 📜 License
This project is for educational purposes only.
© 2025 Parvej Alam.

---

Let me know if you want to include screenshots or a video demo link too!

