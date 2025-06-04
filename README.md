# dbpedia-text-classification-using-Transfomers

# 🧠 DBpedia Transformer Classifier

A Transformer-based text classification project using the [DBpedia CSV dataset](https://www.cs.cmu.edu/~zhiliny/char-cnn/) to predict semantic categories like Company, Film, Artist, and more. Built using PyTorch and spaCy, this model classifies short textual descriptions extracted from Wikipedia.

---

## 📦 Dataset

The dataset is a structured version of Wikipedia content from DBpedia. Each sample includes:
- **Label**: An integer from 0–13 representing the category
- **Title**: The article's title (e.g., "Albert Einstein")
- **Description**: A short abstract or summary of the article

There are:
- **560,000 training samples**
- **70,000 test samples**
- **14 balanced classes**, including:
  - Company
  - EducationalInstitution
  - Artist
  - Athlete
  - OfficeHolder
  - MeanOfTransportation
  - Building
  - NaturalPlace
  - Village
  - Animal
  - Plant
  - Album
  - Film
  - WrittenWork

---

## 🧠 Model Architecture

The model is a custom-built **TransformerClassifier** using PyTorch. Key components include:

- **Token Embeddings**
- **Positional Encoding**
- **Transformer Encoder Layers**
- **Fully Connected Output Layer** (`Linear(embed_dim, num_classes)`)

The model outputs a probability distribution over 14 categories.

---

## 🔧 Setup & Requirements

Install the necessary packages:

```bash
pip install torch pandas numpy matplotlib seaborn spacy gensim
python -m spacy download en_core_web_sm
