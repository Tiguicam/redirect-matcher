#  Matching d'URLs pour Redirection et Migration SEO

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python\&logoColor=white)

---

## 🧩 À propos

Ce script Python permet de faire correspondre automatiquement des URLs 404 avec leurs remplaçantes en 200, dans le cadre :

* d’une **migration de site internet**
* d’un **nettoyage SEO**
* ou d’une **mise en place d’un plan de redirection**

Il combine du **matching Fuzzy** (via RapidFuzz), du **matching sémantique** (via des embeddings), et des **dictionnaires personnalisés** pour améliorer la précision des correspondances.

---

## 🚀 Fonctionnalités

* Nettoyage avancé et normalisation des URLs
* Détection de références produits personnalisables (via regex)
* Support des dictionnaires métiers (genre, produit, couleur, etc.)
* Matching basé sur :

  * Similarité textuelle (Fuzzy)
  * Similarité sémantique (si activée)
  * Pondération par critères métiers
* Interface graphique via Tkinter
* Mode debug manuel pour tester une URL
* Export des résultats au format Excel et debug CSV

---

## 📂 Structure des fichiers

```plaintext
🔹 Redirection_tool.py          # Script principal avec interface
🔹 LICENSE                      # Licence MIT
🔹 README.md                    # Ce fichier :)
🔹 .gitignore                   # Ignore les modèles lourds
🔹 requirements.txt             # Librairies Python nécessaires
```

---

## ⚙️ Pré-requis

* Python 3.7 ou supérieur
* Fichiers Excel contenant les URLs :

  * une feuille nommée `404` avec une colonne `URL_404`
  * une feuille nommée `200` avec une colonne `URL_200`

---

## 📦 Installation

1. Clone ou télécharge le dépôt
2. Installe les dépendances :

```bash
pip install -r requirements.txt
```

> 📌 `sentence-transformers` est **facultatif** :
> si le modèle n’est pas disponible ou si tu n’as pas Internet, le script fonctionne sans la partie sémantique.

---

## 🛠️ Utilisation

1. Lancer le script avec Python :

```bash
python Redirection_tool.py
```

2. Dans l’interface :

   * Sélectionne ton fichier Excel `404` et ton fichier `200`
   * (Optionnel) Importe un dictionnaire Excel ou JSON
   * (Optionnel) Importe un fichier de traduction de mots
   * Ajuste les pondérations, expressions multi-mots, regex, etc.
   * Clique sur **"Lancer le matching"**

3. Le script :

   * nettoie les URLs
   * applique les dictionnaires
   * détecte les références produits
   * et calcule les meilleures correspondances 404 ➔ 200

4. Résultat :

   * Fichier `match_404_200_result.xlsx` généré automatiquement
   * Fichier CSV de debug possible si activé

---

## 📁 Exemple de dictionnaire

Tu peux importer un fichier Excel ou JSON avec ce format :

```plaintext
genre               | couleur              | produit
-------------------|----------------------|---------------------
femme, femmes       | bleu, bleue, blue    | tshirt, tee-shirt
homme, hommes       | rouge, red           | pantalon, pants
```

---

## 💡 Astuces

* Active le **mode debug** pour voir les scores détaillés (embedding, fuzzy, pondérations)
* Tu peux **tester une seule URL manuellement** avec le bouton dédié
* Le champ **regex référence produit** te permet d'adapter le matching à ton format de SKU

---

## 🔧 Améliorations possibles

Pour améliorer encore la qualité des correspondances entre URLs 404 et 200, voici quelques pistes :

* 🧬 Utiliser des modèles d'embedding plus spécialisés (e.g. e-commerce, multilingue)
* 📚 Enrichir les dictionnaires métiers avec plus de synonymes ou variantes
* 🧼 Nettoyer davantage les URLs : stopwords, ponctuation, paramètres inutiles
* 🎯 Personnaliser les pondérations selon le secteur d'activité
* 🔎 Ajouter un filtre de pré-categorisation avant le matching
* 🧪 Ajouter une couche d’évaluation manuelle pour validation

---

## 🔐 Licence

Ce projet est distribué sous la licence **MIT**.
📄 Voir le fichier [LICENSE](./LICENSE)

---

## ✨ Auteur

Développé par **Tigui Camara**
💼 Consultante SEO & créatrice d’outils d’automatisation
📧 Contact : [via GitHub](https://github.com/Tiguicam)

