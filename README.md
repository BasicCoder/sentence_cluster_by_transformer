# Sentence Cluster By Transformer
By using transformers(BERT architecture) for feature extraction, text/sentence clustering is implemented according to the extracted text/sentence semantic features.

## Installation
Use Python3.6 or higher, PyTorch 1.6.0 or higher and transformers v4.6.0 or higher.

Install the *sentence-transformers* with `pip`:
```
pip install -U sentence-transformers
```

## Run
1. Sentence/Paragraphs Embeddings:
```
python sentence_embedding.py
```
For comments in different languages, we need to adopt a multilingual pretrained model to embed different languages into the same semantic space. Only by mapping different languages to the same high-dimensional space can we better carry out subsequent clustering.

These following four multilingual pretrained models can be used for embedding:
* **distiluse-base-multilingual-cased-v1**: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. Supports 15 languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish.
* **distiluse-base-multilingual-cased-v2**: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. This version supports 50+ languages, but performs a bit weaker than the v1 model.
* **paraphrase-multilingual-MiniLM-L12-v2**: Multilingual version of paraphrase-MiniLM-L12-v2, trained on parallel data for 50+ languages.
* **paraphrase-multilingual-mpnet-base-v2**: Multilingual version of paraphrase-mpnet-base-v2, trained on parallel data for 50+ languages.

2. Cluster
```
python cluster.py
```
