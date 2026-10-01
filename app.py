"""Interface Streamlit : tester le classifieur de commentaires citoyens sur un nouveau texte.

Lancement :  streamlit run app.py
Le modèle (model.joblib) est produit par le notebook ; le prétraitement vient de nlp_utils.py.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from nlp_utils import EWE_MARKERS, contains_ewe, normalize_text, preprocess_text

MODEL_PATH = Path(__file__).with_name("model.joblib")
CONFIDENCE_THRESHOLD = 0.60
ICONS = {"Satisfaction": "😊", "Insatisfaction": "😠", "Suggestion": "💡"}
EXAMPLES = {
    "— choisir un exemple —": "",
    "Satisfaction": "Tout s'est bien passé, le personnel était très accueillant.",
    "Insatisfaction": "Mon dossier traîne depuis trois mois sans aucune réponse.",
    "Suggestion": "Il serait utile d'ouvrir un guichet le samedi matin.",
    "Mixte français/éwé (satisfaction)": "Akpe na wò, service très professionnel et efficace.",
    "Mixte français/éwé (insatisfaction)": "Mele via o, service la mègbe trop!",
}


@st.cache_resource
def load_model():
    """Charge le pipeline complet (nettoyage + vectorisation + indices stylistiques + régression logistique)."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Commentaires citoyens — classifieur NLP", page_icon="🗂️")
st.title("🗂️ Classifieur de commentaires citoyens")
st.caption("Satisfaction · Insatisfaction · Suggestion — texte en français, avec insertions éwé/mina possibles.")

if not MODEL_PATH.exists():
    st.error("Fichier `model.joblib` introuvable : exécutez d'abord le notebook (section « Bonus ») pour le générer.")
    st.stop()
try:
    model = load_model()
except Exception as error:  # version de scikit-learn différente, etc.
    st.error(f"Impossible de charger le modèle ({error}). Réinstallez `requirements.txt` ou relancez le notebook.")
    st.stop()

example = st.selectbox("Exemples prêts à l'emploi", list(EXAMPLES))
text = st.text_area("Commentaire à classer", value=EXAMPLES[example], height=120,
                    placeholder="Ex. : Le formulaire en ligne plante à chaque soumission.")

if st.button("Classer le commentaire", type="primary"):
    if not text.strip():
        st.warning("Saisissez un commentaire.")
    else:
        probabilities = pd.Series(model.predict_proba(pd.Series([text]))[0], index=model.classes_)
        label, confidence = probabilities.idxmax(), probabilities.max()

        st.subheader(f"{ICONS[label]} {label}")
        st.metric("Confiance du modèle", f"{confidence:.0%}")
        if confidence < CONFIDENCE_THRESHOLD:
            st.info("Prédiction peu certaine : le commentaire est ambigu ou éloigné des formulations d'entraînement.")
        st.bar_chart(probabilities.rename("Probabilité"))

        if contains_ewe(text):
            found = sorted(set(normalize_text(text).split()) & EWE_MARKERS)
            st.success(f"Termes éwé/mina reconnus : {', '.join(found)}")
        with st.expander("Voir le texte nettoyé transmis au modèle"):
            st.code(preprocess_text(text) or "(vide après nettoyage)")

st.divider()
st.caption("Limite connue : un modèle « sac de mots » gère mal les négations qui s'annulent "
           "(ex. « je n'ai aucune remarque négative »).")
