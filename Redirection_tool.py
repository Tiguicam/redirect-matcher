import pandas as pd
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import re, json

# Pour l'approche embeddings
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    messagebox.showwarning("Warning", "La librairie sentence-transformers n'est pas installée.")
    SentenceTransformer = None
    util = None

DEBUG_MODE = True
global_debug_df = None

# ===================================================================
# Dictionnaires dynamiques
# ===================================================================
ADDITIONAL_DICTS = {}  # { "NOM_DICO": { key: set(synonyms), ... } }

# Table de traduction
TRANSLATION_DICT = {}

# Liste des exceptions pour la normalisation (ex: "cursus" restera inchangé)
NORMALIZATION_EXCEPTIONS = set()

# ===================================================================
# Expressions multi-mots & Stopwords
# ===================================================================
MULTI_WORD_EXPRESSIONS = []
STOPWORDS = set()

# ===================================================================
# Pondérations initiales
# ===================================================================
TOKEN_BONUS_IDENTICAL = 5
TOKEN_PENALTY_EXTRA = -10
EMBEDDING_WEIGHT = 80.0
MIN_SCORE_THRESHOLD = 20

# ===================================================================
# Chargement du Modèle d'Embeddings
# ===================================================================
try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
except Exception as e:
    EMBEDDING_MODEL = None
    print(f"Impossible de charger le modèle d'embedding : {e}")
    print("Assurez-vous d'avoir installé sentence-transformers et d'être connecté à Internet.")

# ===================================================================
# Variables + Regex paramétrable
# ===================================================================
dict_additional_bonus_vars = {}
dict_additional_malus_vars = {}
REFERENCE_PATTERN = r"[A-Za-z]?\d{6,7}-\d{1,2}"
dict_prefilter_vars = {}

# ===================================================================
# Fonctions utilitaires
# ===================================================================

def normalize_word(word: str) -> str:
    """
    Normalise le mot en appliquant la traduction et en retirant le 's'
    final, sauf si le mot figure dans la liste d'exceptions.
    """
    global TRANSLATION_DICT, NORMALIZATION_EXCEPTIONS
    norm_candidate = word.lower().strip()
    if norm_candidate in NORMALIZATION_EXCEPTIONS:
        return TRANSLATION_DICT.get(norm_candidate, norm_candidate)
    if norm_candidate.endswith("s") and norm_candidate not in TRANSLATION_DICT:
        norm_candidate = norm_candidate[:-1]
    return TRANSLATION_DICT.get(norm_candidate, norm_candidate)

def clean_and_normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.lower()
    # Préserver les expressions multi-mots en transformant le tiret en "*"
    for expression in MULTI_WORD_EXPRESSIONS:
        if expression in url:
            url = url.replace(expression, expression.replace("-", "*"))
    # Remplacer les séparateurs par des espaces
    for sep in [".html", ".htm", "/", "-", "_", ".", ",", ":", "!", "?"]:
        url = url.replace(sep, " ")
    url = " ".join(url.split())
    # Normaliser chaque token (traduction, suppression du 's' final, etc.)
    normalized_tokens = []
    for token in url.split():
        token = token.replace("*", "-")
        normalized_tokens.append(normalize_word(token))
    return " ".join(normalized_tokens)

def extract_reference(url: str) -> str:
    if not url:
        return ""
    pattern = REFERENCE_PATTERN
    match = re.search(pattern, url, re.IGNORECASE)
    if match:
        return match.group(0).strip().lower()
    return ""

def load_data(file_404, file_200):
    """
    Charge Excel sheet_name=404 & 200, applique clean & ref.
    Ajoute une colonne REF_DETECTED = OUI/NON.
    """
    df404 = pd.read_excel(file_404, sheet_name="404")
    df200 = pd.read_excel(file_200, sheet_name="200")
    df404["clean_404"] = df404["URL_404"].apply(clean_and_normalize_url)
    df200["clean_200"] = df200["URL_200"].apply(clean_and_normalize_url)
    df404["ref_404"] = df404["URL_404"].apply(extract_reference)
    df200["ref_200"] = df200["URL_200"].apply(extract_reference)
    df404["REF_DETECTED"] = df404["ref_404"].apply(lambda x: "OUI" if x else "NON")
    df200["REF_DETECTED"] = df200["ref_200"].apply(lambda x: "OUI" if x else "NON")
    return df404, df200

def detect_characteristic(text: str, dictionary: dict) -> str:
    """
    Détecte toutes les 'keys' dont au moins un synonyme est présent dans 'text'.
    Retourne une chaîne "key1,key2" si plusieurs clés matchent, ou "" si aucune.
    """
    words = set(text.split())
    matched_keys = []
    for key, synonyms in dictionary.items():
        if words & synonyms:
            matched_keys.append(key)
    return ",".join(matched_keys)

def add_dynamic_characteristics(df, prefix):
    """
    Pour chaque dictionnaire (colonne) dans ADDITIONAL_DICTS,
    crée une colonne ex: 'categorie_404' où sont stockées toutes les clés matchées (séparées par virgule).
    """
    for dico_name, dico_data in ADDITIONAL_DICTS.items():
        col_name = f"{dico_name.lower()}_{prefix}"
        df[col_name] = df[f"clean_{prefix}"].apply(lambda x: detect_characteristic(x, dico_data))
    return df

def get_tokens(text: str) -> set:
    global STOPWORDS
    if not text:
        return set()
    return {t for t in text.split() if t not in STOPWORDS}

def exact_token_bonus(tokens_404: set, tokens_200: set) -> int:
    return TOKEN_BONUS_IDENTICAL * len(tokens_404 & tokens_200)

def extra_tokens_malus(tokens_404: set, tokens_200: set) -> int:
    return TOKEN_PENALTY_EXTRA * len(tokens_200 - tokens_404)

def compute_embedding(text: str):
    if EMBEDDING_MODEL is None:
        return None
    return EMBEDDING_MODEL.encode(text)

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    score = util.cos_sim(vec1, vec2)
    return float(score[0][0])

# Nouvelle fonction : substitution des synonymes avant le fuzzy matching
def apply_synonym_substitution(text: str) -> str:
    """
    Pour chaque token dans le texte, si ce token figure dans l'un des dictionnaires additionnels (dans n'importe quel ensemble de synonymes),
    le remplace par sa clé canonique.
    """
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        replaced = False
        # Pour chaque dictionnaire additionnel
        for dico in ADDITIONAL_DICTS.values():
            for canonical, synonyms in dico.items():
                if token in synonyms:
                    new_tokens.append(canonical)
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            new_tokens.append(token)
    return " ".join(new_tokens)

# On va maintenant utiliser cette fonction dans run_matching pour les 404 ET 200

def debug_manual_url():
    def process_input():
        url = url_entry.get().strip()
        if not url:
            messagebox.showwarning("Champ vide", "Veuillez entrer une URL à tester.")
            return
        cleaned = clean_and_normalize_url(url)
        # Appliquer la substitution sur l'URL testée
        substituted = apply_synonym_substitution(cleaned)
        tokens = get_tokens(substituted)
        ref = extract_reference(url)
        characs = {}
        for dico_name, dico_data in ADDITIONAL_DICTS.items():
            characs[dico_name] = detect_characteristic(substituted, dico_data)
        result = f"🧹 URL nettoyée : {substituted}\n"
        result += f"🔤 Tokens : {', '.join(tokens)}\n"
        result += f"🆔 Référence détectée : {ref or 'Aucune'}\n"
        for k, v in characs.items():
            result += f"📚 {k} → {v or 'Aucune'}\n"
        messagebox.showinfo("Résultat du Debug", result)
    win = tk.Toplevel()
    win.title("Testeur d'URL manuel")
    win.geometry("600x120")
    tk.Label(win, text="Entrez une URL à tester :", font=("Arial", 11)).pack(pady=5)
    url_entry = tk.Entry(win, width=80)
    url_entry.pack(pady=5)
    tk.Button(win, text="Analyser", command=process_input).pack(pady=5)

def run_matching(file_404, file_200, log_widget):
    global DEBUG_MODE, global_debug_df
    global_debug_df = None
    try:
        df404, df200 = load_data(file_404, file_200)
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors du chargement des fichiers: {e}")
        return
    df404 = add_dynamic_characteristics(df404, "404")
    df200 = add_dynamic_characteristics(df200, "200")
    if EMBEDDING_MODEL is not None:
        df404["embedding_404"] = df404["clean_404"].apply(compute_embedding)
        df200["embedding_200"] = df200["clean_200"].apply(compute_embedding)
    else:
        df404["embedding_404"] = None
        df200["embedding_200"] = None

    # IMPORTANT : on applique la substitution par les synonymes sur la version nettoyée,
    # pour que tous les tokens soient déjà convertis en leur forme canonique.
    df404["sub_clean_404"] = df404["clean_404"].apply(apply_synonym_substitution)
    df200["sub_clean_200"] = df200["clean_200"].apply(apply_synonym_substitution)

    best_matches = []
    debug_details = []
    total_404 = len(df404)
    if total_404 == 0:
        messagebox.showwarning("Attention", "Aucune URL 404 trouvée dans le fichier.")
        return
    for i, row_404_data in enumerate(df404.iterrows()):
        _, row_404 = row_404_data
        # Utiliser la version substituée pour le matching
        url_404_sub = row_404["sub_clean_404"]
        embedding_404 = row_404["embedding_404"]
        tokens_404 = get_tokens(url_404_sub)
        ref_404 = row_404["ref_404"]
        same_ref_scored = []
        used_same_ref = False
        if ref_404:
            same_ref_df = df200[df200["ref_200"] == ref_404]
            if not same_ref_df.empty:
                used_same_ref = True
                for _, row_200 in same_ref_df.iterrows():
                    url_200_sub = row_200["sub_clean_200"]
                    tokens_200 = get_tokens(url_200_sub)
                    fuzzy_score = 0
                    score = 0
                    for dico_name_ in ADDITIONAL_DICTS.keys():
                        col_404_name = f"{dico_name_.lower()}_404"
                        col_200_name = f"{dico_name_.lower()}_200"
                        val_404_d = row_404.get(col_404_name, "")
                        val_200_d = row_200.get(col_200_name, "")
                        bonus_strvar = dict_additional_bonus_vars.get(dico_name_)
                        malus_strvar = dict_additional_malus_vars.get(dico_name_)
                        try:
                            bonus_val = int(bonus_strvar.get()) if bonus_strvar else 0
                            malus_val = int(malus_strvar.get()) if malus_strvar else 0
                        except:
                            bonus_val = 0
                            malus_val = 0
                        set_404 = set(val_404_d.split(",")) if val_404_d else set()
                        set_200 = set(val_200_d.split(",")) if val_200_d else set()
                        if set_404 and set_200:
                            if set_404 & set_200:
                                score += bonus_val
                            else:
                                score += (-abs(malus_val))
                        else:
                            score += (-abs(malus_val))
                    score += exact_token_bonus(tokens_404, tokens_200)
                    score += extra_tokens_malus(tokens_404, tokens_200)
                    score += 1000  # Bonus pour référence identique
                    embedding_200 = row_200["embedding_200"]
                    sim_embedding = cosine_similarity(embedding_404, embedding_200)
                    total_score = score + (sim_embedding * EMBEDDING_WEIGHT)
                    if DEBUG_MODE:
                        debug_details.append({
                            "URL_404": row_404["URL_404"],
                            "URL_200_candidate": row_200["URL_200"],
                            "Fuzzy_score": fuzzy_score,
                            "Score_avant_embedding": score,
                            "Similarité_embedding": sim_embedding,
                            "Score_final": total_score
                        })
                    same_ref_scored.append((row_200["URL_200"], total_score))
                same_ref_scored = [m for m in same_ref_scored if m[1] >= MIN_SCORE_THRESHOLD]
                same_ref_scored.sort(key=lambda x: x[1], reverse=True)
        if used_same_ref and len(same_ref_scored) > 0:
            top_match_1 = same_ref_scored[0] if len(same_ref_scored) > 0 else (None, 0)
            top_match_2 = same_ref_scored[1] if len(same_ref_scored) > 1 else (None, 0)
        else:
            df200_filtered = df200.copy()
            for dico_name, dico_data in ADDITIONAL_DICTS.items():
                col_404 = f"{dico_name.lower()}_404"
                col_200 = f"{dico_name.lower()}_200"
                if dict_prefilter_vars.get(dico_name, None) and dict_prefilter_vars[dico_name].get():
                    val_404 = row_404[col_404]
                    if val_404:
                        subset = df200_filtered[df200_filtered[col_200] == val_404]
                        if len(subset) > 0:
                            df200_filtered = subset
            # Ici, on utilise la version substituée pour toutes les candidatures
            candidates_200 = df200_filtered["sub_clean_200"].tolist()
            raw_matches = process.extract(
                url_404_sub,
                candidates_200,
                scorer=fuzz.token_set_ratio,
                limit=100
            )
            detailed_scored = []
            for match_text, fuzzy_score, idx_in_list in raw_matches:
                row_200 = df200_filtered.iloc[idx_in_list]
                url_200_sub = row_200["sub_clean_200"]
                tokens_200 = get_tokens(url_200_sub)
                score = fuzzy_score
                for dico_name_ in ADDITIONAL_DICTS.keys():
                    col_404_name = f"{dico_name_.lower()}_404"
                    col_200_name = f"{dico_name_.lower()}_200"
                    val_404_d = row_404.get(col_404_name, "")
                    val_200_d = row_200.get(col_200_name, "")
                    bonus_strvar = dict_additional_bonus_vars.get(dico_name_)
                    malus_strvar = dict_additional_malus_vars.get(dico_name_)
                    try:
                        bonus_val = int(bonus_strvar.get()) if bonus_strvar else 0
                        malus_val = int(malus_strvar.get()) if malus_strvar else 0
                    except:
                        bonus_val = 0
                        malus_val = 0
                    set_404 = set(val_404_d.split(",")) if val_404_d else set()
                    set_200 = set(val_200_d.split(",")) if val_200_d else set()
                    if set_404 and set_200:
                        if set_404 & set_200:
                            score += bonus_val
                        else:
                            score += (-abs(malus_val))
                    else:
                        score += (-abs(malus_val))
                score += exact_token_bonus(tokens_404, tokens_200)
                score += extra_tokens_malus(tokens_404, tokens_200)
                ref_200 = row_200["ref_200"]
                if ref_404 and ref_200 and ref_404 == ref_200:
                    score += 1000
                embedding_200 = row_200["embedding_200"]
                sim_embedding = cosine_similarity(embedding_404, embedding_200)
                total_score = score + (sim_embedding * EMBEDDING_WEIGHT)
                if DEBUG_MODE:
                    debug_details.append({
                        "URL_404": row_404["URL_404"],
                        "URL_200_candidate": row_200["URL_200"],
                        "Fuzzy_score": fuzzy_score,
                        "Score_avant_embedding": score,
                        "Similarité_embedding": sim_embedding,
                        "Score_final": total_score
                    })
                detailed_scored.append((row_200["URL_200"], total_score))
            filtered_scored = [m for m in detailed_scored if m[1] >= MIN_SCORE_THRESHOLD]
            filtered_scored.sort(key=lambda x: x[1], reverse=True)
            top_match_1 = filtered_scored[0] if len(filtered_scored) > 0 else (None, 0)
            top_match_2 = filtered_scored[1] if len(filtered_scored) > 1 else (None, 0)
        ref_200_top1 = ""
        if top_match_1[0] is not None:
            candidate_row = df200[df200["URL_200"] == top_match_1[0]]
            if not candidate_row.empty:
                ref_200_top1 = candidate_row.iloc[0]["ref_200"]
        exact_ref = "OUI" if ref_404 and ref_200_top1 and ref_404.strip().lower() == ref_200_top1.strip().lower() else "NON"
        best_matches.append({
            "URL_404": row_404["URL_404"],
            "URL_200_top1": top_match_1[0],
            "similarity_top1": top_match_1[1],
            "URL_200_top2": top_match_2[0],
            "similarity_top2": top_match_2[1],
            "EXACT_REF": exact_ref
        })
        percent_done = int(((i + 1) / total_404) * 100)
        progress_label.config(text=f"Progression : {percent_done}%")
        root.update_idletasks()
    df_result = pd.DataFrame(best_matches)
    output_file = "match_404_200_result.xlsx"
    def color_same_ref(row):
        if row["EXACT_REF"] == "OUI":
            return ["background-color: red"] * len(row)
        else:
            return [""] * len(row)
    df_result_styled = df_result.style.apply(color_same_ref, axis=1)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_result_styled.to_excel(writer, index=False)
    if DEBUG_MODE and len(debug_details) > 0:
        global_debug_df = pd.DataFrame(debug_details)
    log_widget.insert(tk.END, f"Matching terminé à 100%. Résultats sauvegardés dans '{output_file}'.\n")
    if DEBUG_MODE and global_debug_df is not None:
        log_widget.insert(tk.END, "Debug en mémoire : cliquez sur 'Exporter Debug CSV' pour le sauvegarder.\n")
    plt.hist(df_result["similarity_top1"], bins=20, alpha=0.7)
    plt.xlabel("Score de Similarité (top1) (Fuzzy + Sémantique)")
    plt.ylabel("Nombre d'URLs")
    plt.title(f"Distribution des Scores (Min = {MIN_SCORE_THRESHOLD})")
    plt.show()

def enforce_bonus_malus_limits(*args):
    try:
        b_token = int(var_token_bonus.get() or 0)
        m_token = int(var_token_malus.get() or 0)
    except:
        return
    add_bonus_sum = 0
    add_malus_sum = 0
    for dico_name in dict_additional_bonus_vars:
        try:
            b_val = int(dict_additional_bonus_vars[dico_name].get() or 0)
            add_bonus_sum += b_val
        except:
            pass
    for dico_name in dict_additional_malus_vars:
        try:
            m_val = int(dict_additional_malus_vars[dico_name].get() or 0)
            add_malus_sum += m_val
        except:
            pass
    total_bonus = b_token + add_bonus_sum
    total_malus = m_token + add_malus_sum
    if total_bonus > 100:
        messagebox.showinfo("Limite bonus dépassée", f"La somme des bonus ({total_bonus}) dépasse 100.\nVous pouvez continuer malgré tout.")
    if total_malus > 100:
        messagebox.showinfo("Limite malus dépassée", f"La somme des malus ({total_malus}) dépasse 100.\nVous pouvez continuer malgré tout.")
    lbl_bonus_sum.config(text=f"Total Bonus : {total_bonus}/100")
    lbl_malus_sum.config(text=f"Total Malus : {total_malus}/100")

def generate_dictionary_template():
    columns = ["genre", "couleur", "produit"]
    data = [
        {
            "genre": "femme, femme, femmes, woman",
            "couleur": "bleu, bleu, bleue, bleus, blue",
            "produit": "tshirt, tshirt, tee-shirt, ts"
        },
        {
            "genre": "homme, homme, hommes, man",
            "couleur": "rouge, rouge, rouges, red",
            "produit": "pantalon, pant, pants"
        }
    ]
    df = pd.DataFrame(data, columns=columns)
    out_file = "dictionnaire_template.xlsx"
    try:
        df.to_excel(out_file, index=False)
        messagebox.showinfo("Template Dictionnaire", f"Template créé : {out_file}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de créer le template: {e}")

def import_dictionary_excel():
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if not filename:
        return
    global ADDITIONAL_DICTS
    ADDITIONAL_DICTS.clear()
    try:
        df = pd.read_excel(filename)
        for dico_name in df.columns:
            dico_content = {}
            for cell_value in df[dico_name].dropna():
                items = [normalize_word(it.strip()) for it in str(cell_value).split(",") if it.strip()]
                if not items:
                    continue
                key = items[0]
                synonyms = set(items)
                if key in dico_content:
                    dico_content[key] = dico_content[key].union(synonyms)
                else:
                    dico_content[key] = synonyms
            ADDITIONAL_DICTS[dico_name] = dico_content
        create_additional_dictionary_fields()
        messagebox.showinfo("Import Dictionnaire", f"Dictionnaires importés depuis {filename}.\nDicos détectés: {list(ADDITIONAL_DICTS.keys())}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de lire le fichier: {e}")

def import_dictionary_json():
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if not filename:
        return
    global ADDITIONAL_DICTS
    ADDITIONAL_DICTS.clear()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            config = json.load(f)
        for dico_name, dict_items in config.items():
            dico_content = {}
            for key, synonyms in dict_items.items():
                norm_key = normalize_word(key.strip())
                norm_synonyms = {normalize_word(s.strip()) for s in synonyms}
                dico_content[norm_key] = norm_synonyms
            ADDITIONAL_DICTS[dico_name] = dico_content
        create_additional_dictionary_fields()
        messagebox.showinfo("Import Dictionnaire JSON", f"Dictionnaires importés depuis {filename}.\nDicos détectés: {list(ADDITIONAL_DICTS.keys())}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de l'importation du dictionnaire : {e}")

def generate_translation_template():
    data = {
        "SOURCE": ["belt", "belts", "sac"],
        "TRANSLATION": ["ceinture", "ceinture", "bag"]
    }
    df = pd.DataFrame(data)
    out_file = "translation_template.xlsx"
    try:
        df.to_excel(out_file, index=False)
        messagebox.showinfo("Template Traduction", f"Template créé : {out_file}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de créer le template: {e}")

def import_translation_excel():
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if not filename:
        return
    global TRANSLATION_DICT
    TRANSLATION_DICT.clear()
    try:
        df = pd.read_excel(filename)
        if not all(col in df.columns for col in ["SOURCE", "TRANSLATION"]):
            messagebox.showwarning("Format invalide", "Le fichier doit contenir 'SOURCE' et 'TRANSLATION'.")
            return
        for _, row in df.iterrows():
            src = normalize_word(str(row["SOURCE"]).strip())
            tra = normalize_word(str(row["TRANSLATION"]).strip())
            TRANSLATION_DICT[src] = tra
        messagebox.showinfo("Import Traduction", f"Table de traduction importée depuis {filename}.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de lire le fichier: {e}")

def import_translation_json():
    filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if not filename:
        return
    global TRANSLATION_DICT
    TRANSLATION_DICT.clear()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            trans = json.load(f)
        for k, v in trans.items():
            key_norm = normalize_word(str(k).strip())
            val_norm = normalize_word(str(v).strip())
            TRANSLATION_DICT[key_norm] = val_norm
        messagebox.showinfo("Import Traduction JSON", f"La table de traduction a été importée depuis {filename}.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de l'importation de la table de traduction : {e}")

root = tk.Tk()
root.title("Matching URL (Fuzzy + Sémantique + Dicos dynamiques)")

files_frame = tk.LabelFrame(root, text="Fichiers Excel (404 et 200)")
files_frame.pack(fill="x", padx=5, pady=5)

tk.Label(files_frame, text="Fichier 404:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
file404_entry = tk.Entry(files_frame, width=50)
file404_entry.grid(row=0, column=1, padx=5, pady=2)

def browse_file(entry):
    filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

tk.Button(files_frame, text="Parcourir", command=lambda: browse_file(file404_entry)).grid(row=0, column=2, padx=5, pady=2)
tk.Label(files_frame, text="Fichier 200:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
file200_entry = tk.Entry(files_frame, width=50)
file200_entry.grid(row=1, column=1, padx=5, pady=2)
tk.Button(files_frame, text="Parcourir", command=lambda: browse_file(file200_entry)).grid(row=1, column=2, padx=5, pady=2)

import_frame = tk.LabelFrame(root, text="Dictionnaires, Traduction")
import_frame.pack(fill="x", padx=5, pady=5)

tk.Button(import_frame, text="Générer Template Dictionnaire", command=generate_dictionary_template).grid(row=0, column=0, padx=5, pady=5)
tk.Button(import_frame, text="Importer Dictionnaire Excel", command=import_dictionary_excel).grid(row=0, column=1, padx=5, pady=5)
tk.Button(import_frame, text="Importer Dictionnaire JSON", command=import_dictionary_json).grid(row=0, column=2, padx=5, pady=5)
tk.Button(import_frame, text="Générer Template Traduction", command=generate_translation_template).grid(row=1, column=0, padx=5, pady=5)
tk.Button(import_frame, text="Importer Traduction Excel", command=import_translation_excel).grid(row=1, column=1, padx=5, pady=5)
tk.Button(import_frame, text="Importer Traduction JSON", command=import_translation_json).grid(row=1, column=2, padx=5, pady=5)

params_frame = tk.LabelFrame(root, text="Pondérations & Score")
params_frame.pack(fill="x", padx=5, pady=5)

var_token_bonus = tk.StringVar(value="0")
var_token_malus = tk.StringVar(value=str(abs(TOKEN_PENALTY_EXTRA)))
var_embedding_weight = tk.StringVar(value=str(EMBEDDING_WEIGHT))
var_min_score_threshold = tk.StringVar(value=str(MIN_SCORE_THRESHOLD))

var_token_bonus.trace("w", enforce_bonus_malus_limits)
var_token_malus.trace("w", enforce_bonus_malus_limits)

row_idx = 0
tk.Label(params_frame, text="Token (bonus):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
tk.Entry(params_frame, width=6, textvariable=var_token_bonus).grid(row=row_idx, column=1, padx=5, pady=2)
tk.Label(params_frame, text="Token (malus):").grid(row=row_idx, column=2, sticky="e", padx=5, pady=2)
tk.Entry(params_frame, width=6, textvariable=var_token_malus).grid(row=row_idx, column=3, padx=5, pady=2)

row_idx += 1
tk.Label(params_frame, text="Poids Sémantique:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
tk.Entry(params_frame, width=6, textvariable=var_embedding_weight).grid(row=row_idx, column=1, padx=5, pady=2)
tk.Label(params_frame, text="Score Minimum:").grid(row=row_idx, column=2, sticky="e", padx=5, pady=2)
tk.Entry(params_frame, width=6, textvariable=var_min_score_threshold).grid(row=row_idx, column=3, padx=5, pady=2)

lbl_bonus_sum = tk.Label(params_frame, text="Total Bonus : 0/100")
lbl_bonus_sum.grid(row=row_idx, column=4, padx=5, pady=5, sticky="e")
lbl_malus_sum = tk.Label(params_frame, text="Total Malus : 0/100")
lbl_malus_sum.grid(row=row_idx, column=5, padx=5, pady=5, sticky="e")

dynamic_dico_frame = tk.LabelFrame(root, text="Dictionnaires additionnels : Pondérations + Préfiltrage")
dynamic_dico_frame.pack(fill="x", padx=5, pady=5)

def create_additional_dictionary_fields():
    for widget in dynamic_dico_frame.winfo_children():
        widget.destroy()
    dict_additional_bonus_vars.clear()
    dict_additional_malus_vars.clear()
    dict_prefilter_vars.clear()
    row = 0
    for dico_name in ADDITIONAL_DICTS.keys():
        var_prefilter = tk.BooleanVar(value=False)
        dict_prefilter_vars[dico_name] = var_prefilter
        chk_prefilter = tk.Checkbutton(dynamic_dico_frame, text=f"Préfiltrer sur {dico_name}", variable=var_prefilter)
        chk_prefilter.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        tk.Label(dynamic_dico_frame, text=f"{dico_name} (bonus):").grid(row=row, column=1, sticky="e", padx=5, pady=2)
        bonus_var = tk.StringVar(value="0")
        dict_additional_bonus_vars[dico_name] = bonus_var
        e_bonus = tk.Entry(dynamic_dico_frame, width=5, textvariable=bonus_var)
        e_bonus.grid(row=row, column=2, padx=5, pady=2)
        bonus_var.trace("w", enforce_bonus_malus_limits)
        tk.Label(dynamic_dico_frame, text=f"{dico_name} (malus):").grid(row=row, column=3, sticky="e", padx=5, pady=2)
        malus_var = tk.StringVar(value="0")
        dict_additional_malus_vars[dico_name] = malus_var
        e_malus = tk.Entry(dynamic_dico_frame, width=5, textvariable=malus_var)
        e_malus.grid(row=row, column=4, padx=5, pady=2)
        malus_var.trace("w", enforce_bonus_malus_limits)
        row += 1

norm_frame = tk.LabelFrame(root, text="Paramètres de Normalisation")
norm_frame.pack(fill="x", padx=5, pady=5)

tk.Label(norm_frame, text="Expressions multi-mots (séparées par virgule) :").grid(row=0, column=0, padx=5, pady=2, sticky="e")
multi_expr_entry = tk.Entry(norm_frame, width=60)
multi_expr_entry.insert(0, "")
multi_expr_entry.grid(row=0, column=1, padx=5, pady=2)
tk.Label(norm_frame, text="Stopwords (séparés par virgule) :").grid(row=1, column=0, padx=5, pady=2, sticky="e")
stopwords_entry = tk.Entry(norm_frame, width=60)
stopwords_entry.insert(0, "")
stopwords_entry.grid(row=1, column=1, padx=5, pady=2)
tk.Label(norm_frame, text="Regex référence produit :").grid(row=2, column=0, padx=5, pady=2, sticky="e")
var_reference_pattern = tk.StringVar(value=REFERENCE_PATTERN)
regex_entry = tk.Entry(norm_frame, width=60, textvariable=var_reference_pattern)
regex_entry.grid(row=2, column=1, padx=5, pady=2)
tk.Label(norm_frame, text="Exceptions de normalisation (séparées par virgule) :").grid(row=3, column=0, padx=5, pady=2, sticky="e")
exceptions_entry = tk.Entry(norm_frame, width=60)
exceptions_entry.insert(0, "")  # Par exemple, saisir "cursus" pour ne pas retirer le 's'
exceptions_entry.grid(row=3, column=1, padx=5, pady=2)

button_frame = tk.Frame(root)
button_frame.pack(fill="x", padx=5, pady=5)

def update_parameters():
    global TOKEN_BONUS_IDENTICAL, TOKEN_PENALTY_EXTRA, MULTI_WORD_EXPRESSIONS, STOPWORDS, EMBEDDING_WEIGHT, REFERENCE_PATTERN, MIN_SCORE_THRESHOLD, NORMALIZATION_EXCEPTIONS
    try:
        TOKEN_BONUS_IDENTICAL = int(var_token_bonus.get() or 0)
        TOKEN_PENALTY_EXTRA = -abs(int(var_token_malus.get() or 0))
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs entières pour le bonus/malus.")
        return
    expr = multi_expr_entry.get()
    MULTI_WORD_EXPRESSIONS[:] = [e.strip() for e in expr.split(",") if e.strip()]
    sw_text = stopwords_entry.get()
    new_stopwords = {s.strip().lower() for s in sw_text.split(",") if s.strip()}
    STOPWORDS.clear()
    STOPWORDS.update(new_stopwords)
    REFERENCE_PATTERN = var_reference_pattern.get()
    try:
        new_embed_weight = float(var_embedding_weight.get())
    except ValueError:
        new_embed_weight = 80.0
    EMBEDDING_WEIGHT = new_embed_weight
    try:
        new_threshold = float(var_min_score_threshold.get())
    except ValueError:
        new_threshold = 20
    MIN_SCORE_THRESHOLD = new_threshold
    exceptions_text = exceptions_entry.get()
    NORMALIZATION_EXCEPTIONS.clear()
    NORMALIZATION_EXCEPTIONS.update({e.strip().lower() for e in exceptions_text.split(",") if e.strip()})
    messagebox.showinfo("Paramètres", "Pondérations et paramètres mis à jour.")

def run_interface():
    update_parameters()
    file404 = file404_entry.get()
    file200 = file200_entry.get()
    if not file404 or not file200:
        messagebox.showwarning("Attention", "Veuillez sélectionner les deux fichiers Excel (404 et 200).")
        return
    log_text.insert(tk.END, "Lancement du matching (Fuzzy + Sémantique + Dicos dynamiques)...\n")
    run_matching(file404, file200, log_text)

run_button = tk.Button(button_frame, text="Lancer le Matching", command=run_interface)
run_button.pack(side="left", padx=5, pady=5)

def export_debug_csv():
    global DEBUG_MODE, global_debug_df
    if not DEBUG_MODE:
        messagebox.showinfo("Info", "Le mode debug est désactivé, aucun debug disponible.")
        return
    if global_debug_df is None:
        messagebox.showwarning("Attention", "Aucun debug n'a été généré ou le matching n'a pas encore été lancé.")
        return
    filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if filename:
        try:
            global_debug_df.to_csv(filename, index=False)
            messagebox.showinfo("Export Debug", f"Fichier debug exporté : {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer le debug : {e}")

debug_export_button = tk.Button(button_frame, text="Exporter Debug CSV", command=export_debug_csv)
debug_export_button.pack(side="left", padx=10, pady=5)

debug_manual_button = tk.Button(button_frame, text="Tester une URL manuellement", command=debug_manual_url)
debug_manual_button.pack(side="left", padx=10, pady=5)

log_frame = tk.LabelFrame(root, text="Logs")
log_frame.pack(fill="both", expand=True, padx=5, pady=5)
log_text = scrolledtext.ScrolledText(log_frame, height=10)
log_text.pack(fill="both", expand=True, padx=5, pady=5)
progress_label = tk.Label(log_frame, text="Progression : 0%")
progress_label.pack(anchor="e", padx=5, pady=5)

enforce_bonus_malus_limits()

root.mainloop()
