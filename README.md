# Classification de commentaires citoyens (NLP) — Test technique TAISS 2026 / Afriklang

- **Données** : 150 commentaires courts (≈ 9 mots), 3 classes équilibrées (Satisfaction / Insatisfaction / Suggestion), dont 3 mêlant français et éwé/mina.
- **Nettoyage** (`nlp_utils.py`) : minuscules, accents neutralisés (lettres éwé conservées), élisions développées, ponctuation et chiffres supprimés. Les **stopwords sont une liste maison** qui *conserve* négations, pronoms de 1ʳᵉ personne, intensifs et marqueurs de modalité : +0,024 de F1 mesuré par rapport à l'absence de filtrage.
- **Éwé/mina** : tokens conservés + mini-glossaire sûr (`akpe` → merci, `nyuie` → bien) ; les 3 stratégies (garder / traduire / supprimer) sont comparées, sans écart mesurable (3 cas sur 150).
- **Vectorisation** : TF-IDF sur les mots, comparé à `CountVectorizer`, aux bigrammes et aux n-grammes de caractères.
- **Indices stylistiques** (issus de l'exploration) : chiffre, conditionnel, infinitif de consigne, 1ʳᵉ personne, calculés sur le texte brut et ajoutés au TF-IDF.
- **Modèles** : régression logistique, SVM linéaire, Naive Bayes, Random Forest ; split 80/20 stratifié (`random_state=42`) + validation croisée 5×5, car le test ne compte que 30 exemples.
- **Résultats** : TF-IDF seul ≈ **0,70** de F1 macro (CV) ; avec les indices ≈ **0,86 ± 0,06**. Modèle final (régression logistique) : **29/30 sur le test (accuracy = F1 macro = 0,967)** ; la valeur de référence reste la CV.
- **Erreurs** : 23 des 24 erreurs « out-of-fold » sont des confusions Satisfaction ↔ Insatisfaction, dues à la négation (*« je n'ai aucune remarque négative »*) ; les Suggestions sont quasi parfaitement détectées.
- **Limites** : corpus très petit et très formaté (les indices exploitent ses conventions), négation mal gérée par le sac de mots, éwé/mina non validable statistiquement avec 3 exemples.
- **Pistes** : fine-tuning CamemBERT / XLM-R (ou embeddings LaBSE), marquage de la portée de la négation, davantage de données réelles et glossaire éwé validé par des locuteurs natifs.
- **Bonus** : modèle sauvegardé (`model.joblib`), interface Streamlit (`app.py`), traitement dédié des commentaires mixtes français/éwé.

## Reproduire

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab nlp_classification_citoyens.ipynb          # « Run All » : régénère nlp_utils.py et model.joblib
streamlit run app.py                                   # interface de test
```
