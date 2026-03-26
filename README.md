# Atelier Big Data ? Traitement par l?intelligence artificielle

Reconnaissance de **chiffres** et de **lettres** manuscrites avec un r�seau de neurones (MLP), entra�n� sur les jeux de donn�es du projet.

## Pr�requis

- **Python 3.10** ou plus r�cent
- Les dossiers de donn�es :
  - `archive/` (chiffres : `numbers.csv` + images dans `archive/numbers/`)
  - `Handwritten letters.v1i.folder/train/` (lettres : sous-dossiers par classe 0-9, A-Z, a-z)

## Installation

1. **Cloner ou t�l�charger** le projet, puis ouvrir un terminal dans le dossier du projet.

2. **Installer les d�pendances** :

   ```powershell
   pip install -r requirements.txt
   ```

## Commandes

### Entra�ner le mod�le (chiffres + lettres)

```powershell
python train_cnn.py
```

- Charge les donn�es depuis `archive` et `Handwritten letters.v1i.folder/train/`.
- Entra�ne un MLP (62 classes : 0-9, A-Z, a-z).
- Enregistre le mod�le dans `model_chiffres_lettres.joblib`, les noms de classes dans `model_classes.json`, et la matrice de confusion dans `confusion_matrix.png`.

### Pr�dire sur une image ou un dossier

**Une image :**

```powershell
python predict_digit.py chemin/vers/image.png
python predict_digit.py chemin/vers/image.jpg
```

**Toutes les images d?un dossier :**

```powershell
python predict_digit.py "Handwritten letters.v1i.folder/train/A"
python predict_digit.py archive/numbers/chars74k_png/GoodImg/Sample5
```

Le script affiche, pour chaque image, le caract�re pr�dit et la confiance (ex. : `pr�dit : 'A' (confiance : 95.00 %)`).

## lancer l'interface graphique
```powershell
flask run --debug --no-reload
```

## Structure du projet

| Fichier / Dossier | R�le |
|-------------------|------|
| `train_cnn.py` | Entra�nement du r�seau de neurones (chiffres + lettres) |
| `predict_digit.py` | Pr�diction du caract�re (chiffre ou lettre) sur une image |
| `requirements.txt` | D�pendances Python |
| `model_chiffres_lettres.joblib` | Mod�le entra�n� (cr�� apr�s `train_cnn.py`) |
| `model_classes.json` | Liste des 62 classes (cr�� apr�s `train_cnn.py`) |
| `confusion_matrix.png` | Matrice de confusion (cr��e apr�s `train_cnn.py`) |
| `archive/` | Donn�es chiffres (CSV + images) |
| `Handwritten letters.v1i.folder/train/` | Donn�es lettres (sous-dossiers par caract�re) |

## Param�tres d?entra�nement

Dans `train_cnn.py`, vous pouvez modifier :

- **`MAX_SAMPLES_ARCHIVE`** : nombre max d?images utilis�es depuis l?archive (par ex. `20000`, ou `None` pour tout utiliser).
- **`MAX_SAMPLES_LETTERS`** : idem pour le dossier des lettres (par ex. `15000`, ou `None`).
- **`VALIDATION_SPLIT`** : part des donn�es utilis�e pour la validation (ex. `0.20` = 20 %).
- **`max_iter`** : nombre d?�poques d?entra�nement du MLP (ex. `50`).

## D�pendances principales

- `scikit-learn` ? r�seau de neurones (MLPClassifier)
- `pandas`, `numpy` ? donn�es et calculs
- `Pillow` ? chargement d?images (PNG, JPG)
- `matplotlib` ? matrice de confusion
- `joblib` ? sauvegarde du mod�le
