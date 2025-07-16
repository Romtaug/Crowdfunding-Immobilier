#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py – Streamlit + scraping LPB (v2024-07)
• Imports & configuration Streamlit
• Connexion Selenium (login LPB)
• Helpers communs
• parse() EXACTEMENT comme dans le batch, avec meilleure détection Ville/Département
"""

from __future__ import annotations

# ╭───────────────  Imports  ───────────────╮
import io, re, time, pickle, pathlib
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st
import joblib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import streamlit as st, os
# ╰─────────────────────────────────────────╯


# ╭────────────  Streamlit config  ─────────────╮
st.set_page_config(
    page_title="La Première Brique | API",
    page_icon="🧱",
    layout="wide",
)
# ╰─────────────────────────────────────────────╯
email_auth = st.secrets["auth"]["email"]
password_auth = st.secrets["auth"]["password"]

# ╭────────── 1. Selenium (login LPB) ──────────╮
EMAIL = st.secrets["auth"]["email"]
PASSWORD = st.secrets["auth"]["password"]
TIMEOUT, HEADLESS = 5, True

@st.cache_resource(show_spinner="⌛ Connexion à LPB…")
def get_driver() -> webdriver.Edge:
    opts = EdgeOptions()
    opts.use_chromium = True
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    if HEADLESS:
        opts.add_argument("--headless=new")

    drv   = webdriver.Edge(
        service=EdgeService(EdgeChromiumDriverManager().install()),
        options=opts,
    )
    wait  = WebDriverWait(drv, TIMEOUT)

    drv.get("https://app.lapremierebrique.fr/fr/users/sign_in")

    # cookies
    try:
        wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(),'Tout autoriser')]"))
        ).click()
    except Exception:
        pass                                                    # déjà accepté

    # login
    wait.until(EC.presence_of_element_located((By.NAME, "user[email]")))
    drv.find_element(By.NAME, "user[email]").send_keys(EMAIL)
    drv.find_element(By.NAME, "user[password]").send_keys(PASSWORD)
    drv.find_element(By.NAME, "commit").click()
    return drv
# ╰─────────────────────────────────────────────╯


# ╭──────── 2. Helpers communs batch/app ───────╮
text_or  = lambda n, d="": n.get_text(" ", strip=True) if n else d
str2eur  = lambda s: float(re.sub(r"[^\d,]", "", s).replace(",", ".") or 0)
SPACE_MAP = str.maketrans({" ": " ", "\u202F": " ", "\u2009": " "})
unspace   = lambda s: s.translate(SPACE_MAP)
ENT_REGEX = re.compile(r"\b(SAS|SCI|SARL|SA)\s+[A-Z0-9À-Ÿ'’\- ]{3,}")

COLS = [
    "id","extraction","url","nom","Ville","Département","Typologie","Entreprise",
    "Notation_LPB","Sûreté","Mode_rémunération","Billet_min","Taux_pub",
    "Intérêt_contractuel","Durée","Collecte","Maximum","Objectif","Restant",
    "Progression","Investisseurs","Ouverture","Remboursement","Clôture_prog",
    "SIREN","Equipe","Présentation","Elements_financiers","Calendrier",
    "Meta_description","Nb_commentaires","URL_replay","Galerie_photos",
    "Image","Statut",
]

def blank_row(pid:int)->dict[str,str]:
    """Toutes les colonnes existent d’emblée, valeurs vides."""
    return {c: (pid if c == "id" else "") for c in COLS}
# ╰─────────────────────────────────────────────╯


# ╭───────── 3. parse() – identique batch ──────╮
def recap_stats(soup: BeautifulSoup) -> dict[str, str]:
    return {
        text_or(td.find("span", class_="text-dark-grey")).lower():
        text_or(td.find("span", class_="fw-bold"))
        for td in soup.select("table tbody tr td.bg-body-secondary")
    }

def data_layer(soup: BeautifulSoup) -> tuple[str, str]:
    typ = tgt = ""
    for sc in soup.find_all("script"):
        js = sc.text
        if not typ and (m := re.search(r'"typology"\s*:\s*"([^"]+)"', js)):
            typ = m.group(1)
        if not tgt and (m := re.search(r'"targeted_amount"\s*:\s*"([^"]+)"', js)):
            tgt = m.group(1)
        if typ and tgt:
            break
    return typ, tgt

def section(soup: BeautifulSoup, pattern: str) -> str:
    h = soup.find("h3", string=re.compile(pattern, re.I)); out = []
    while h and (h := h.find_next_sibling()):
        if h.name == "h3":
            break
        out.append(text_or(h))
    return " ".join(out).strip()

def parse(html: str) -> dict[str, str] | None:
    soup = BeautifulSoup(html, "html.parser")
    og   = soup.select_one('[property="og:url"]')
    if not og or not (mid := re.search(r"/projects/(\d+)$", og["content"])):
        return None

    pid  = int(mid.group(1))
    row  = blank_row(pid)
    row["extraction"] = "trouvé"
    row["url"]        = og["content"]

    # ---- Titres & sous-titre --------------------------------------------
    row["nom"]   = text_or(soup.h1)
    subtitle     = text_or(soup.select_one("h2.h6"))

    # ---- Ville / Département  (exactement le batch) ---------------------
    city = dep = ""
    # ①  Forme « … à Hyères (83) »
    m = re.search(r"à\s+([^,(]+?)\s*\((\d{2,3})\)", subtitle)
    if m:
        city, dep = m.group(1).strip(), m.group(2)
    else:
        # ②  Sinon on prend le dernier « (NN) »
        m = re.search(r"([^,(]+?)\s*\((\d{2,3})\)\s*$", subtitle)
        if m:
            city, dep = m.group(1).strip(), m.group(2)

    row["Ville"]       = city
    row["Département"] = dep

    # ---- Tableau récapitulatif ------------------------------------------
    stats = recap_stats(soup)
    row["Collecte"]      = stats.get("collectés", stats.get("collecté", ""))
    row["Maximum"]       = stats.get("maximum", "")
    row["Investisseurs"] = stats.get("investisseurs", stats.get("investisseur", ""))

    # Durée & intérêt contractuel
    for g, b in stats.items():
        if (m := re.search(r"\b(\d+)\s*mois\b", unspace(g))):
            row["Durée"] = f"{m.group(1)} mois"; row["Intérêt_contractuel"] = b; break
    if not row["Durée"]:
        if (m := re.search(r"durée[^0-9]*(\d+)\s*mois", unspace(subtitle), re.I)):
            row["Durée"] = f"{m.group(1)} mois"

    # Progression
    if (bar := soup.select_one(".progressbar .bar[style]")) and \
       (m := re.search(r"([\d.]+)", bar["style"])):
        row["Progression"] = f"{float(m.group()):.0f} %"

    # Typologie & objectif
    row["Typologie"] = text_or(soup.select_one("span.pill-type"))
    typ_json, obj_json = data_layer(soup)
    if not row["Typologie"]:
        row["Typologie"] = typ_json
    row["Objectif"] = obj_json

    # Présentation longue
    presentation = section(soup, "Présentation")
    if (m := ENT_REGEX.search(presentation)):
        row["Entreprise"] = m.group(0).strip()

    # Billet mini & taux pub
    if (m := re.search(r"Investissez dès ([\d ]+)€", subtitle)):
        row["Billet_min"] = f"{m.group(1).replace(' ', '')} €"
    if (m := re.search(r"au taux de ([\d,.% ]+)", subtitle)):
        row["Taux_pub"] = m.group(1).strip()

    # Note LPB
    row["Notation_LPB"] = text_or(soup.select_one("span.pill"))

    # Sûreté
    if (h := soup.find("h3", string=re.compile("sûretés", re.I))) and \
       (li := h.find_next("li", class_="border-top")):
        row["Sûreté"] = text_or(li)

    # Mode de rémunération
    row["Mode_rémunération"] = text_or(
        soup.find("li", string=re.compile("Mode de rémunération", re.I)))

    # Sections longues
    row["Elements_financiers"] = section(soup, "Eléments financiers")
    row["Calendrier"]          = section(soup, "Calendrier prévisionnel")
    row["Présentation"]        = presentation

    # Équipe
    if (h := soup.find("h3", string=re.compile("Équipe projet", re.I))) and \
       (ul := h.find_next("ul")):
        row["Equipe"] = "; ".join(text_or(li) for li in ul.find_all("li"))

    # SIREN
    if (m := re.search(r"\b\d{9}\b", row["Elements_financiers"] or presentation)):
        row["SIREN"] = m.group()

    # Clôture prévisionnelle
    if (m := re.search(r"Clôture[^:]*:\s*([^.;]+)", row["Elements_financiers"], re.I)):
        row["Clôture_prog"] = m.group(1).strip()

    # Ouverture / remboursement
    if (a := soup.select_one('a[href*="calendar/render"][href*="dates="]')) and \
       (m := re.search(r"dates=(\d{8}T\d{6})", a["href"])):
        row["Ouverture"] = datetime.strptime(
            m.group(1), "%Y%m%dT%H%M%S").strftime("%d/%m/%Y %H:%M")
    if (t := soup.find("time", datetime=True)):
        row["Remboursement"] = t["datetime"]

    # Image & statut
    if (cover := soup.select_one("#projectCover")):
        if (m := re.search(r"url\((.*?)\)", cover.get("style", ""))):
            row["Image"] = m.group(1)
        if (st := cover.select_one("img.project-status-show")):
            row["Statut"] = st["src"]

    # Meta-description, commentaires, YouTube, galerie
    row["Meta_description"] = text_or(
        soup.select_one('[name="description"],[property="og:description"]'))
    row["Nb_commentaires"] = text_or(soup.select_one("#comments span.count"))
    if (ifr := soup.select_one('iframe[src*="youtube.com/embed"]')):
        row["URL_replay"] = ifr["src"]
    row["Galerie_photos"] = ", ".join(
        img["src"] for img in soup.select("#projectGallery img[src]"))

    # Restant à collecter
    if row["Collecte"] and row["Objectif"]:
        row["Restant"] = f"{str2eur(row['Objectif']) - str2eur(row['Collecte']):,.0f} €".replace(",", " ")

    return row


def scrape_one_url(url: str) -> pd.DataFrame | None:
    drv  = get_driver()
    wait = WebDriverWait(drv, TIMEOUT)
    try:
        drv.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
        time.sleep(2)                                           # micro-pause : JS
    except Exception as e:
        st.error(f"Selenium : {e}")
        return None

    row = parse(drv.page_source)
    if row is None:
        st.error("URL non reconnue comme projet LPB.")
        return None
    return pd.DataFrame([row], columns=COLS)
# ╰─────────────────────────────────────────────╯

# ───────── paramètres alignés sur le training
TARGET_COL   = "Statut"
DROP_COLS    = ["id", "url", "extraction", "Image", "nom", "Progression", "Maximum"]
DATE_COLS    = ("Ouverture", "Clôture_prog")
RARE_THRESH  = 5

ROOT      = pathlib.Path(__file__).parent
OUT_DIR   = ROOT / "outputs"
IMG_DIR   = OUT_DIR / "images"

MODEL_PKL  = OUT_DIR / "xgb_model.pkl"
LABELS_PKL = OUT_DIR / "labels.pkl"
DATA_XLSX  = ROOT / "Projets.xlsx"          # pour prédiction par ID

PNG = {
    "Accuracy global":           IMG_DIR / "accuracy_kpi.png",
    "Matrice de confusion": IMG_DIR / "confusion_matrix.png",
    "Top 15 importances":   IMG_DIR / "top15_features.png",
    "Accuracy par classe":  IMG_DIR / "accuracy_par_classe.png",
    "Arbre simplifié":      IMG_DIR / "arbre_decision_simplifie.png",
    "Learning curve":       IMG_DIR / "xgb_learning_curve.png",
}

# ───────── pré-traitement identique au pipeline
def preprocess(df_raw: pd.DataFrame, model) -> pd.DataFrame:
    df = df_raw.copy()

    # drop colonnes
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # dates → année
    for col in DATE_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.year.astype("Int64")

    # numériques : imput moyenne
    num_cols = df.select_dtypes(include=["number", "Int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # catégorielles : rares & NA
    cat_cols = [c for c in df.columns if c not in num_cols and c != TARGET_COL]
    for c in cat_cols:
        s = df[c].astype(str)
        rare = s.value_counts()[lambda s_: s_ < RARE_THRESH].index
        s = s.where(~s.isin(rare), "RARE").where(~s.isna(), "NA")
        df[c] = pd.Categorical(s).codes.astype("int32")

    # retire target le cas échéant
    if TARGET_COL in df:
        df = df.drop(columns=[TARGET_COL])

    # aligne exactement sur les colonnes du modèle
    return df.reindex(columns=model.feature_names_in_, fill_value=0)

# ───────── caches Streamlit
@st.cache_resource(show_spinner="⏳ Chargement du modèle…")
def load_model():
    return joblib.load(MODEL_PKL)

@st.cache_resource(show_spinner="⏳ Chargement des labels…")
def load_label_map() -> dict[int, str]:
    if LABELS_PKL.exists():
        with open(LABELS_PKL, "rb") as f:
            classes = pickle.load(f)
        return {i: lab for i, lab in enumerate(classes)}
    st.warning("labels.pkl manquant – fallback ordre alphabétique.")
    labels = sorted(pd.read_excel(DATA_XLSX)[TARGET_COL].dropna().unique())
    return {i: lab for i, lab in enumerate(labels)}

@st.cache_data(show_spinner="⏳ Lecture du fichier Excel…")
def load_data():
    return pd.read_excel(DATA_XLSX)

model      = load_model()
LABEL_MAP  = load_label_map()
df_all     = load_data()


# ───────── interface
st.sidebar.title("🧱 Menu")
page = st.sidebar.radio("Navigation",
                        ("Mode d'emploi", "Scraping de données", "Batch prédictions", "Résultats du modèle"))

# A. Résultats + visuels (vertical, taille réduite)
if page == "Résultats du modèle":
    st.header("📊 Résultats du modèle")

    for title, path in PNG.items():
        st.caption(title)
        if path.exists():
            st.image(path)   # ✅ plus de use_column_width
        else:
            st.warning(f"{path.name} manquante")

# ------------------------------------------------------------------
elif page == "Mode d'emploi":
    st.header("ℹ️ Mode d’emploi")

    st.markdown(
        """
### Pourquoi cette appli ?

> Aider à anticiper le **destin d’un projet**  
> *Sera-t-il financé ? remboursé ? ou en échec de financement ?*  
> Le modèle s’appuie exclusivement sur les **critères publics** du dossier immobilier : localisation, sûretés, durée de remboursement, taux annualisé, typologie, etc.

| Étape | Ce qu’on fait | Sortie |
|-------|---------------|--------|
| **1️⃣ Scraping LPB** | Selenium ouvre le projet, se logue, accepte les cookies puis récupère l’HTML. `parse()` (BeautifulSoup + regex) extrait **35 champs** : titre, Ville/Département, sûretés, tableau financier, images… | `pd.DataFrame` 1 ligne |
| **2️⃣ Pré-traitement** | Transformation **DataPrep** (100 % pandas)<br>• Dates → année<br>• Numériques : imputation moyenne<br>• Catégorielles : seuil « RARE » (5) + codes `int32`<br>• Départements gardés comme catégorie (« 05 », « 83 », …) | Matrice `X` prête pour le modèle |
| **3️⃣ Modèle XGBoost** | `Optuna` cherche les meilleurs hyperparamètres de l'arbre de décision.<br>Objectif : `multi:softprob` → classes **Financé / Remboursé / Échec**.<br>Early-Stopping : 50 rounds. | `xgb_model.pkl` |
| **4️⃣ Évaluation** | Accuracy globale + par classe, matrice de confusion, courbe d’apprentissage, importance des features. | 6 PNG dans *outputs/images/* |
| **5️⃣ Prédictions** | • *Scraping de données* : prédiction pour **1 URL**<br>• *Batch prédictions* : CSV/XLSX de plusieurs projets ✚ colonne **Statut prédit** | Fichier enrichi (download) |

### Scraping en bref
1. `get_driver()` lance Edge (Chromium).  
2. Cookies acceptés, login automatique.  
3. Dès que la première balise du site est visible, on passe la page à `BeautifulSoup`
4. `parse()` pioche l’ID, la ville, le département, les montants, la barre de progression, le JSON *data-layer*, etc.  
5. On retourne un `DataFrame` normalisé : **même fonction** pour l’interface et les scripts batch.

### Modèle ML
* **DataPrep** maintient la cohérence (moyennes, catégories, encodage) entre entraînement et production.  
* `OptunaSearchCV` explore 8 hyperparamètres ⇒ ~600 fits (10 folds). Score : `f1_macro`
* Artefacts sauvegardés : `preprocess.pkl` `xgb_model.pkl` `labels.pkl`

---

### 📂 **Où trouver les fichiers du projet complet ?**

Les dossiers du projet sont disponibles ici : **[GitHub – Crowdfunding Immobilier](https://github.com/Romtaug/Crowdfunding-Immobilier)**

- `1 - Scraping - Python Selenium` : récupère les données des projets sélectionnés de La Première Brique.
- `2 - Dashboard - Power Query BI` : Power BI d'analyse des projets de la start-up.
- `3 - API Scoring - Python Streamlit` : interface prédictive pour scorer les projets et savoir s'ils vont être financés ou non.

### 📊 Comment mettre à jour le dashboard Power BI :

1. Lancez le scraping d’un ou plusieurs projets via cette interface.
2. Actualisez le fichier Excel macro query dans le dossier dashboard.
3. Ouvrez Power BI et cliquez sur **"Actualiser tout"**.
4. Les données seront mises à jour automatiquement dans le tableau de bord.

### 🤖 Comment mettre à jour le modèle de prédiction :

1. Mettez à jour le fichier Excel de nettoyage (prétraitement).
2. Exécutez le fichier `.ipynb` du modèle XGBoost.
3. Le modèle sera mis à jour, l’interface se lancera et sera fonctionnelle automatiquement.
""",
        unsafe_allow_html=True,
    )


elif page == "Scraping de données":
    st.header("🕵️‍♂️ Scraping de données")

    st.markdown(
        """ 

### Comment scraper un projet La Première Brique ?

1. Allez sur la page officielle des projets : [https://app.lapremierebrique.fr/fr/projects](https://app.lapremierebrique.fr/fr/projects)
2. Choisissez un projet dans la liste.
3. Copiez l’URL du projet sélectionné (par exemple : [https://app.lapremierebrique.fr/fr/projects/20](https://app.lapremierebrique.fr/fr/projects/20))
4. Collez-la ci-dessous pour lancer le scraping :
"""
    )

    input_url = st.text_input(
        "Collez l’URL du projet La Première Brique :",
        value="https://app.lapremierebrique.fr/fr/projects/20",
    )

    if st.button("Scraper"):
        if not input_url:
            st.error("Merci de fournir une URL.")
            st.stop()

        # Validation stricte de l'URL LPB
        pattern = r"^https://app\.lapremierebrique\.fr/fr/projects/\d+$"
        if not re.match(pattern, input_url.strip()):
            st.error("❌ L’URL fournie ne correspond pas au format attendu d’un projet LPB.")
            st.stop()

        try:
            # 1) Scraping brut -------------------------------------------------
            df_scraped = scrape_one_url(input_url)
            if df_scraped is None:
                st.error("Le projet n’a pas pu être extrait. Vérifiez l’URL ou réessayez plus tard.")
                st.stop()

            st.subheader("Colonnes récupérées (scraping brut)")
            st.dataframe(
                df_scraped.T,
                use_container_width=True,
                height=min(600, 22 * df_scraped.shape[1]),
            )
        except Exception as e:
            st.error(f"❌ Une erreur est survenue pendant le scraping, testez le programme en local ou réessayez plus tard.")


# C. Batch prédictions
else:
    st.header("🤖 Batch prédictions")

    st.markdown("""
### Prédire plusieurs projets en une fois

Chargez un fichier `.xlsx` ou `.csv` contenant plusieurs projets LPB à scorer automatiquement. Vous obtiendrez grâce à notre modèle bien entraîné la classe prédite et la fiabilité de prédiction du projet.

Voici le fichier issu du scraping avec les projets récents (après le changement de la structure du code) :
""")

    import os
    import pathlib

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Projets.xlsx")

    # 🔹 Bouton de téléchargement du fichier d'exemple
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger le fichier d'exemple Projets.xlsx",
                data=f,
                file_name="Projets.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ Le fichier `Projets.xlsx` n’a pas été trouvé dans le dossier courant.")

    # 🔹 Uploader (drag and drop possible)
    up = st.file_uploader("Déposez ici un fichier à scorer (ou laissez vide pour charger Projets.xlsx)", type=["xlsx", "csv"])

    # 🔹 Lecture du fichier (priorité au drag & drop)
    df_new = None
    if up is not None:
        ext = pathlib.Path(up.name).suffix.lower()
        try:
            df_new = pd.read_excel(up) if ext == ".xlsx" else pd.read_csv(up, sep=None, engine="python")
            st.info("Fichier chargé depuis le drag-and-drop.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            st.stop()
    elif os.path.exists(file_path):
        try:
            df_new = pd.read_excel(file_path)
            st.info("Le fichier `Projets.xlsx` a été chargé automatiquement depuis le dossier local.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier local : {e}")
            st.stop()
    else:
        st.warning("❌ Aucun fichier trouvé. Veuillez déposer un fichier ou ajouter `Projets.xlsx` dans le dossier.")
        st.stop()

    # 🔍 Aperçu
    st.write("🔍 Aperçu du fichier :", df_new.head())

    # 🔄 Prétraitement
    X_clean = preprocess(df_new, model)

    # 🧠 Prédiction + score de fiabilité
    proba = model.predict_proba(X_clean)
    preds = proba.argmax(axis=1)
    confiance = proba.max(axis=1) * 100  # Score en %

    df_new["Statut prédit"] = [LABEL_MAP.get(int(i), "Inconnu") for i in preds]
    df_new["Fiabilité financement (%)"] = confiance.round(1)

    # ✅ Affichage
    st.success("✅ Prédictions terminées")
    st.dataframe(df_new, use_container_width=True)

    # 💾 Téléchargements
    csv_bytes = df_new.to_csv(index=False).encode("utf-8")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_new.to_excel(writer, index=False)

    # 3 colonnes vides à gauche, 2 au centre, 3 à droite pour centrer visuellement
    col1, col2, col3, col_csv, col_xlsx, col6, col7, col8 = st.columns([1, 1, 1, 2, 2, 1, 1, 1])

    with col_csv:
        st.download_button(
            "📥 Télécharger en CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            key="dl_csv",
        )

    with col_xlsx:
        st.download_button(
            "📥 Télécharger en Excel",
            data=buf.getvalue(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
        )
