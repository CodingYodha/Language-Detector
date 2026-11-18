# Language-Detector

A small Streamlit app that tells whether a piece of text is **Kannada** or **Marathi**.

## How it works

1. The input text is turned into features with a saved TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
2. A saved classifier (`language_detection_model.pkl`) gives a probability for each language.
3. If the top probability is below 0.75, the app says the text is neither Kannada nor Marathi. Otherwise it shows the predicted language and its confidence.

## Results

[TODO: add test accuracy. The training notebook in `notebooks/` is empty, so add the training code and the result]

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The folder layout (`data/`, `src/`, `reports/`, …) comes from the Cookiecutter Data Science template. Most of those folders are still empty placeholders.
