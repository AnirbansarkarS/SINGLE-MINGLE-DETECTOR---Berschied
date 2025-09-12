# SINGLE-MINGLE-DETECTOR---Berschied
(❤️ Relationship Status Predictor)

A fun **Machine Learning + Streamlit app** that predicts whether a person is **Single** or **Committed** based on lifestyle and social behavior patterns.

The project uses a **Decision Tree / Random Forest model** trained on a **realistic dataset** of features like Instagram usage, number of male/female friends, late-night talks, gym frequency, etc.
The frontend is built using **Streamlit** where users answer a set of questions and get predictions + visual insights (pie chart).

---

## 🚀 Features

* Interactive **Streamlit frontend** with 10+ lifestyle questions.
* Predicts **Single / Committed** status in a fun, non-direct way.
* Shows results in **graphical format (Pie chart)**.
* Backend ML model built using **scikit-learn**.
* Dataset balanced for realistic results (50-50 split).

---

## 📂 File Structure

```
relationship-predictor/
│
├── data/
│   └── realistic_relationship_dataset_balanced.csv   # Dataset
│
├── models/
│   └── decision_tree.pkl    # Trained ML model
│
├── app.py                   # Streamlit frontend
├── train.py                 # Training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
...
---

## ⚙️ Installation

Clone the repo and set up the environment:

```bash
git clone https://github.com/AnirbansarkarS/SINGLE-MINGLE-DETECTOR---Berschied.git
cd relationship-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

To retrain the model with the dataset:

```bash
python train.py
```

This will generate `models/decision_tree.pkl`.

---

## 🎨 Running the App

Launch Streamlit:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 📊 Example Questions

* How many **male friends** do you have?
* How many **female friends** do you have?
* How many hours per day do you spend on **Instagram**?
* How often do you **go to parties** in a month?
* Do you have frequent **late-night talks**?
* How many hours per week do you **hit the gym**?
* How many times do you **hang out in coffee shops** monthly?

---

## 🖼️ Output

* Prediction: **Single** / **Committed**
* Confidence shown as a **Pie Chart**.

---

## ⚠️ Disclaimer

This project is for **educational and fun purposes only**.
The predictions are based on **synthetic data + cheat features** and should not be taken seriously.

---

✨ Built with ❤️ using **Python, scikit-learn, and Streamlit**
