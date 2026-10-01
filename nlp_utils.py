"""Prétraitement et indices stylistiques pour les commentaires citoyens (français + insertions éwé/mina)."""
import re
import unicodedata

import numpy as np

# =============================================================================
# 1. Stratégie éwé/mina
# =============================================================================
# Lettres propres à l'éwé : elles ne se décomposent pas en NFKD, donc elles sont conservées telles quelles.
EWE_LETTERS = "ɛɔɖŋƒɣʋ"
# Marqueurs éwé repérés dans le corpus (comparés après normalisation, donc sans accents).
EWE_MARKERS = {"mele", "via", "megbe", "yevu", "nyuie", "hafi", "akpe", "wo"}
# Mini-glossaire : uniquement les traductions certaines (expressions d'abord, puis mots isolés).
EWE_PHRASES = {"akpe na wo": "merci"}
EWE_WORDS = {"akpe": "merci", "nyuie": "bien"}

# =============================================================================
# 2. Stopwords français (liste maison, justifiée dans le notebook)
# =============================================================================
STOPWORDS_FR = {
    # articles et déterminants
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux", "ce", "cet", "cette", "ces",
    "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    # pronoms sans valeur discriminante, relatifs, démonstratifs
    "tu", "elle", "ils", "elles", "te", "se", "lui", "qui", "que", "dont", "ou",
    "celui", "celle", "ceux", "cela", "ca", "ceci",
    # prépositions et conjonctions de liaison
    "a", "dans", "par", "sur", "sous", "avec", "chez", "entre", "vers", "depuis", "pendant", "avant", "apres",
    "et", "ni", "mais", "donc", "or", "car", "si", "comme", "lors", "afin",
    # formes conjuguées d'être/avoir (le conditionnel « serait » est volontairement conservé)
    "est", "es", "suis", "sommes", "etes", "sont", "etais", "etait", "etaient", "ete", "etre",
    "ai", "as", "avons", "avez", "ont", "avait", "avaient", "avoir", "eu",
    # quelques adverbes sans valeur discriminante
    "alors", "aussi", "ainsi", "encore", "deja", "meme", "autre", "autres", "quelque", "quelques",
}
# Mots volontairement CONSERVÉS, bien que présents dans la plupart des listes standard :
#  - négations : ne, pas, non, aucun(e), jamais, rien, sans, personne -> polarité (Insatisfaction)
#  - intensité : tres, trop, plus, moins, bien, tout, tous
#  - première personne et « il » impersonnel : je, me, mon, ma, mes, on, nous, vous, il, en, y
#    -> récits d'expérience vécue (Satisfaction/Insatisfaction) vs formulations impersonnelles (Suggestion)
#  - modalité/conditionnel : serait, faudrait, faut, pour -> marqueurs de Suggestion
GARDES_CRITIQUES = {
    "ne", "pas", "non", "aucun", "aucune", "jamais", "rien", "sans", "personne",
    "tres", "trop", "plus", "moins", "bien", "tout", "tous",
    "je", "me", "mon", "ma", "mes", "on", "nous", "vous", "il", "en", "y",
    "pour", "serait", "faut", "faudrait",
}
assert not (STOPWORDS_FR & GARDES_CRITIQUES), "Un mot critique ne doit pas figurer dans les stopwords"

# Liste d'affichage (nuages de mots, top 15) : on masque EN PLUS les mots-outils conservés pour le modèle
# (pronoms, « ne », « pour »…) et le terme omniprésent « service », afin que les graphiques montrent les mots de contenu.
STOPWORDS_VISU = STOPWORDS_FR | {"je", "me", "mon", "ma", "mes", "on", "nous", "vous", "il", "en", "y", "ne", "pour",
                                 "tout", "tous", "plus", "moins", "service", "services"}

# Élisions restituées sous leur forme pleine : on préserve « je », « me » et surtout la négation « ne ».
ELISIONS = {"j": "je", "m": "me", "t": "te", "s": "se", "n": "ne", "l": "le", "c": "ce", "d": "de"}


# =============================================================================
# 3. Fonctions de nettoyage
# =============================================================================
def fold_accents(text: str) -> str:
    """Supprime les accents français (é -> e) en conservant les lettres éwé (ɛ, ɔ, ɖ, ŋ, ƒ, ɣ, ʋ)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Minuscules, accents neutralisés, élisions développées, ponctuation/chiffres/caractères spéciaux supprimés."""
    text = fold_accents(text.lower().replace("’", "'"))
    text = re.sub(r"\bqu'", "que ", text)
    text = re.sub(r"\b([jmtsnlcd])'", lambda m: ELISIONS[m.group(1)] + " ", text)   # j'ai -> je ai
    text = re.sub(rf"[^a-z{EWE_LETTERS}\s]", " ", text)   # ne garde que lettres latines + lettres éwé
    return re.sub(r"\s+", " ", text).strip()


def apply_ewe_strategy(text: str, strategy: str) -> str:
    """strategy = 'keep' (tokens éwé conservés), 'translate' (glossaire sûr) ou 'remove' (supprimés)."""
    if strategy == "keep":
        return text
    if strategy == "translate":
        for phrase, french in EWE_PHRASES.items():
            text = text.replace(phrase, french)
        return " ".join(EWE_WORDS.get(token, token) for token in text.split())
    if strategy == "remove":
        return " ".join(token for token in text.split() if token not in EWE_MARKERS)
    raise ValueError(f"Stratégie inconnue : {strategy}")


def preprocess_text(text: str, ewe_strategy: str = "keep", remove_stopwords: bool = True,
                    stopwords=STOPWORDS_FR, min_len: int = 2) -> str:
    """Pipeline complet : normalisation -> stratégie éwé -> tokenisation -> filtrage -> texte nettoyé."""
    tokens = apply_ewe_strategy(normalize_text(text), ewe_strategy).split()
    tokens = [t for t in tokens if len(t) >= min_len and not (remove_stopwords and t in stopwords)]
    return " ".join(tokens)


def contains_ewe(text: str) -> bool:
    """Vrai si le commentaire contient au moins un marqueur éwé/mina connu."""
    return any(token in EWE_MARKERS for token in normalize_text(text).split())


# =============================================================================
# 4. Utilitaires pour les pipelines scikit-learn (entrée = textes bruts)
# =============================================================================
STYLE_FEATURE_NAMES = ["style_chiffre", "style_conditionnel", "style_debut_infinitif", "style_premiere_personne"]


def clean_batch(texts, **kwargs):
    """Applique preprocess_text à une collection de textes (compatible FunctionTransformer)."""
    return [preprocess_text(text, **kwargs) for text in texts]


# Mots qui suivent typiquement un infinitif de consigne (« Créer UNE FAQ », « Mettre EN place », « Publier RÉGULIÈREMENT… »)
_AFTER_INFINITIVE = {"le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "au", "aux", "en", "a", "pour", "avec",
                     "sur", "dans", "par", "ses", "ces", "nos", "vos", "leurs", "mieux", "davantage", "plus", "toujours"}


def starts_with_infinitive(words) -> bool:
    """Formule de proposition : 1er mot à l'infinitif (-er/-ir/-re) SUIVI d'un déterminant, d'une préposition ou d'un adverbe en -ment.

    Le second critère évite les faux positifs sur les noms en -er/-re placés en tête de plainte
    (« Dossier perdu », « Formulaire illisible », « Papier… »), que la seule terminaison ne distingue pas des infinitifs.
    """
    if len(words) < 2 or not re.search(r"(?:er|ir|re)$", fold_accents(words[0])):
        return False
    second = fold_accents(words[1])
    return second in _AFTER_INFINITIVE or second.endswith("ment")


def linguistic_features(texts):
    """Indices stylistiques binaires calculés sur le texte BRUT (avant suppression des chiffres/accents).

    Colonnes (cf. STYLE_FEATURE_NAMES) :
      - contient un chiffre (durées, quantités : typique des plaintes chiffrées)
      - contient un verbe au conditionnel (-rait / -raient : « serait », « permettrait »…)
      - commence par un infinitif de consigne (« Créer une… », « Mettre en place… »)
      - contient un pronom de première personne (récit d'une expérience vécue)
    """
    rows = []
    for text in texts:
        lowered = text.lower().replace("’", "'")
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœɛɔɖŋƒɣʋ]+", lowered)    # sans apostrophe : « d'avoir » -> d, avoir
        rows.append([
            int(bool(re.search(r"\d", text))),
            int(bool(re.search(r"\w+rai(?:t|ent)\b", lowered))),
            int(starts_with_infinitive(words)),
            int(bool(re.search(r"\b(?:je|j'|m'|mon|ma|mes|me)\b", lowered))),
        ])
    return np.array(rows, dtype=float)
