# CreditWise Loan System

A Streamlit app that predicts whether a loan application is likely to be
**Approved** or **Rejected**, based on a Logistic Regression model trained
on SecureTrust Bank's historical loan data.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app (loads `model.pkl` and serves the UI) |
| `train.py` | Trains the model from `loan_approval_data.csv` and saves `model.pkl` |
| `model.pkl` | Pre-trained pipeline (preprocessing + Logistic Regression) — already included |
| `loan_approval_data.csv` | Training data (only needed if you re-run `train.py`) |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Re-train the model (optional)

If you update the dataset and want to retrain:

```bash
python train.py
```

This regenerates `model.pkl`. Then just rerun the app — no code changes needed.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **public or private GitHub repo**. Make sure
   `app.py`, `model.pkl`, and `requirements.txt` are all committed.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file
   path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL.

> You do **not** need to include `loan_approval_data.csv` in the deployed
> repo unless you want the option to retrain from the cloud — the app only
> needs `model.pkl` to serve predictions.

## Deploy elsewhere (Render, Railway, HF Spaces, etc.)

Any platform that can run:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

will work, as long as `requirements.txt` is installed first.

## Notes

- The model is a **Logistic Regression** pipeline (chosen automatically in
  `train.py` by comparing F1 score against KNN and Naive Bayes).
- Predictions are decision-support only — they are not a substitute for
  final human loan-officer review.
