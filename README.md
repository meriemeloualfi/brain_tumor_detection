# 🧠 Détection de Tumeurs Cérébrales par CNN

Classification automatique d'images IRM cérébrales en 4 classes grâce au **Transfer Learning** (VGG19, avec comparaison ResNet50 et MobileNetV2 possible).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Statut](https://img.shields.io/badge/Statut-Projet%20acad%C3%A9mique-lightgrey)

---

## 📌 Objectif

Développer un système capable de distinguer, à partir d'une image IRM, entre :

| Classe | Description |
|---|---|
| `glioma` | Tumeur gliale |
| `meningioma` | Méningiome (tumeur des méninges) |
| `pituitary` | Tumeur hypophysaire (adénome) |
| `notumor` | Cerveau sain |

> ⚠️ **Avertissement** : ce projet est à but pédagogique. Il ne constitue pas un dispositif médical et ne doit pas servir à établir un diagnostic.

---

## 📂 Dataset

- **Source** : [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Utilisé dans le notebook** : 5 600 images d'entraînement + 1 600 images de test, classes parfaitement équilibrées (1 400 / 400 images par classe)
- **Découpage** : 85 % entraînement (4 760 images) / 15 % validation (840 images) / test séparé (1 600 images)

Structure attendue :

```
brain_tumor_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🛠️ Méthodologie

1. **Prétraitement** : redimensionnement 224×224, normalisation (`rescale=1/255`)
2. **Data augmentation** : rotation (±15°), décalages (10 %), zoom (10 %), cisaillement, variation de luminosité, flip horizontal
3. **Architecture** : backbone VGG19 pré-entraîné sur ImageNet + tête personnalisée
   `GlobalAveragePooling2D → BatchNorm → Dense(512) → Dropout → Dense(256) → Dropout → Softmax(4)`
   (régularisation L2 sur les couches denses)
4. **Entraînement en 2 phases**
   - **Phase 1 – Feature Extraction** : backbone gelé, `lr = 1e-3`, 15 époques max
   - **Phase 2 – Fine-Tuning** : dégel des 4 dernières couches du backbone, `lr = 1e-5`, 20 époques max
5. **Callbacks** : `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`
6. **Évaluation** : rapport de classification, matrice de confusion, courbes ROC (One-vs-Rest)
7. **Explicabilité** : cartes de chaleur **Grad-CAM** (dernière couche convolutive `block5_conv4`)

---

## 📊 Résultats

Résultats obtenus lors de l'exécution du notebook (VGG19 fine-tuné) :

| Métrique | Valeur |
|---|---|
| Meilleure accuracy de validation | **95,71 %** |
| Accuracy sur le jeu de test | **91,69 %** |
| F1-score macro (test) | **0,9149** |
| F1-score pondéré (test) | **0,9149** |

> ℹ️ Le tableau comparatif VGG19 / ResNet50 / MobileNetV2 présent dans le notebook contient des valeurs **illustratives** (à remplacer par vos propres mesures après entraînement de chaque modèle).

Figures générées automatiquement : `sample_images.png`, `class_distribution.png`, `augmentation_examples.png`, `training_history.png`, `confusion_matrix.png`, `roc_curves.png`, `gradcam_example.png`.

---

## 🚀 Installation et utilisation

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd <nom-du-depot>

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset (nécessite ~/.kaggle/kaggle.json)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p brain_tumor_dataset

# 5. Lancer le notebook
jupyter notebook brain_tumor_detection.ipynb
```

Adaptez `DATASET_PATH` dans le notebook si votre dataset est ailleurs. Une exécution sur **GPU** est fortement recommandée (sur CPU, une époque de fine-tuning prend ~15 minutes).

### Prédire avec Grad-CAM

```python
display_gradcam('./brain_tumor_dataset/Testing/glioma/Te-gl_0001.jpg', model_vgg19)
```

### Sauvegarder le modèle

```python
save_model_artifacts(model_vgg19)   # SavedModel + .h5 + métadonnées JSON
```

---

## 📁 Structure du projet

```
.
├── brain_tumor_detection.ipynb   # Notebook principal
├── requirements.txt              # Dépendances
├── brain_tumor_dataset/          # Données (non versionnées)
└── README.md
```

---

## 🧰 Technologies

TensorFlow / Keras · scikit-learn · OpenCV · NumPy · Pandas · Matplotlib · Seaborn · tf-keras-vis

---

## 🔮 Pistes d'amélioration

- Entraîner et comparer réellement ResNet50 et MobileNetV2
- Dégeler davantage de couches lors du fine-tuning
- Migrer vers le format `.keras` (le format `.h5` est legacy)
- Ajouter une interface de démonstration (Streamlit / Gradio)
- Validation croisée et analyse des erreurs entre glioma et méningiome

---

## 👤 Auteur

**Meriem EL OUALFI**
