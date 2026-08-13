# Fake News Detection with BERT

Un projet de classification binaire (fake vs. real news) basé sur le fine-tuning de **BERT** (`bert-base-uncased`), avec une baseline **TF-IDF + Régression Logistique** pour comparaison.

##  Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle), composé de deux fichiers CSV :
- `Fake.csv` — articles de fake news
- `True.csv` — articles de vraies news (Reuters)

##  Nettoyage des données

Le dataset brut contient plusieurs artefacts qui rendent la tâche trop facile pour un modèle (fuite d'information) :
- Le tag `(Reuters)` présent presque exclusivement dans les vrais articles
- Le préfixe `"CITY (Reuters) - "` en début de texte
- Des apostrophes manquantes (`Here s what` → `Here's what`)
- Des mentions `"Featured image via ..."`, des URLs et des liens Twitter (`pic.twitter.com/...`)
- Des guillemets/apostrophes typographiques non normalisés

Une fonction `clean_text()` corrige ces artefacts avant l'entraînement.

##  Pipeline

1. **Chargement & nettoyage** des deux CSV
2. **Préparation** : fusion `title + text`, labellisation (`0` = vrai, `1` = fake), mélange aléatoire
3. **Split** : 70 % train / 15 % validation / 15 % test (stratifié)
4. **Tokenisation** avec `BertTokenizer` (`bert-base-uncased`, `max_length=256`)
5. **Fine-tuning** de `BertForSequenceClassification` avec :
   - Dropout renforcé (0.3 au lieu de 0.1) pour limiter le surapprentissage
   - `learning_rate=2e-5`, `weight_decay=0.05`, `warmup_ratio=0.1`
   - `EarlyStoppingCallback` (patience = 2) sur le F1-score
6. **Évaluation** sur le jeu de test
7. **Baseline de comparaison** : TF-IDF (5000 features) + Régression Logistique

##  Résultats

### BERT fine-tuné (jeu de validation)

| Métrique  | Score  |
|-----------|--------|
| Accuracy  | 0.988  |
| Precision | 1.000  |
| Recall    | 0.977  |
| F1        | 0.988  |

### Baseline TF-IDF + Régression Logistique (jeu de test)

| Métrique  | Score |
|-----------|-------|
| Accuracy  | 0.98  |
| F1 (avg)  | 0.98  |

>  Les deux approches atteignent des scores très proches de 99 %, même après nettoyage des artefacts évidents. Cela suggère que le dataset reste "trop facile" (les modèles apprennent probablement encore des indices stylistiques/structurels propres à la source plutôt que du raisonnement factuel), un point à garder en tête pour toute utilisation en conditions réelles.

##  Stack technique

- Python, Pandas
- PyTorch
- Hugging Face `transformers` (BERT) et `datasets`
- scikit-learn (baseline, métriques, split)

##  Utilisation

```bash
pip install transformers datasets torch scikit-learn pandas
```

Le notebook `fakenewsproject.ipynb` (initialement développé sur Kaggle) contient l'ensemble du pipeline, de l'exploration des données jusqu'à l'entraînement et l'évaluation. Adaptez les chemins `/kaggle/input/...` et `/kaggle/working/...` à votre environnement local si nécessaire.

Le modèle fine-tuné est sauvegardé dans un dossier `bert_fake_news_model/` (poids + tokenizer), rechargeable avec :

```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("bert_fake_news_model")
tokenizer = BertTokenizer.from_pretrained("bert_fake_news_model")
```

##  Structure du projet

```
.
├── fakenewsproject.ipynb   # Notebook principal (exploration, nettoyage, entraînement, évaluation)
└── README.md
```


