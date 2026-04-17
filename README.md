# CS-4063 NLP Assignment 2: Neural Urdu NLP Pipeline

This project implements Assignment 2 for CS-4063 Natural Language Processing. It extends a BBC Urdu NLP pipeline with vector-space representations, neural word embeddings, sequence labelling, and topic classification models implemented from scratch in PyTorch.

The main work is contained in `i23-2508_Assignment2_DS-B.ipynb`. The notebook includes the full pipeline, saved intermediate artifacts, trained model checkpoints, plots, and written analysis.

## Project Overview

The assignment follows the restrictions in `Assignment 2.pdf`: no pretrained models, no Gensim, no HuggingFace, and no use of PyTorch's built-in Transformer modules such as `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`.

Implemented components include:

- Corpus loading and Urdu text encoding repair
- Vocabulary construction capped at 10,000 tokens
- TF-IDF weighted term-document matrix
- PPMI co-occurrence matrix and t-SNE visualization
- Skip-gram Word2Vec model from scratch in PyTorch
- Embedding evaluation with nearest neighbours, analogies, and MRR
- Rule-assisted POS and NER dataset creation in CoNLL format
- BiLSTM POS tagger with frozen and fine-tuned embeddings
- BiLSTM NER tagger with CRF and non-CRF decoding comparisons
- From-scratch Transformer encoder for topic classification
- BiLSTM topic classifier baseline
- Training curves, confusion matrices, attention heatmaps, and final written comparison

## Folder Structure

```text
.
|-- Assignment 2.pdf
|-- README.md
|-- cleaned.txt
|-- raw.txt
|-- metadata.json
|-- i23-2508_Assignment2_DS-B.ipynb
|-- data/
|   |-- pos_train.conll
|   |-- pos_test.conll
|   |-- ner_train.conll
|   |-- ner_test.conll
|   |-- ner_crf_predictions.conlleval
|   `-- ner_nocrf_predictions.conlleval
|-- embeddings/
|   |-- word2idx.json
|   |-- tfidf_matrix.npy
|   |-- tfidf_matrix_sparse.npz
|   |-- ppmi_matrix.npy
|   |-- ppmi_matrix_sparse.npz
|   `-- embeddings_w2v.npy
|-- models/
|   |-- bilstm_pos.pt
|   |-- bilstm_pos_frozen.pt
|   |-- bilstm_ner.pt
|   `-- transformer_cls.pt
`-- plots/
    |-- w2v_loss_curve.png
    |-- ppmi_tsne.png
    |-- mrr_comparison.png
    |-- pos_training_curves.png
    |-- pos_confusion_matrix.png
    |-- transformer_training_curves.png
    |-- transformer_confusion_matrix.png
    |-- attn_heatmap_article1.png
    |-- attn_heatmap_article5.png
    `-- attn_heatmap_article7.png
```

## Input Files

The core input files are:

- `raw.txt`: original Urdu article corpus
- `cleaned.txt`: cleaned Urdu article corpus used as the primary training text
- `metadata.json`: article titles and publication dates, used for weak topic labels
- `Assignment 2.pdf`: official assignment specification

The corpus contains 300 articles. The notebook repairs encoding issues before downstream processing so Urdu text is handled in readable UTF-8 form.

## Environment

The notebook was written for Python 3 and PyTorch. It can run on CPU, but GPU is strongly recommended for the neural sections.

Required Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib seaborn tqdm torch conlleval
```

The notebook automatically detects CUDA and can use multiple GPUs through `nn.DataParallel` when available.

## How to Run

1. Open the notebook:

   ```bash
   jupyter notebook i23-2508_Assignment2_DS-B.ipynb
   ```

2. Run all cells from top to bottom.

3. The notebook will create or refresh these output folders as needed:

   - `embeddings/`
   - `models/`
   - `data/`
   - `plots/`

4. Review the final comparison section near the end of the notebook for the written analysis and model comparison table.

## Generated Artifacts

Current generated artifacts include:

| Artifact | Shape / Contents |
| --- | --- |
| `embeddings/word2idx.json` | 10,000 vocabulary entries |
| `embeddings/embeddings_w2v.npy` | Word2Vec embeddings, shape `(10000, 100)` |
| `embeddings/tfidf_matrix.npy` | Dense TF-IDF matrix, shape `(300, 10000)` |
| `embeddings/tfidf_matrix_sparse.npz` | Sparse TF-IDF matrix |
| `embeddings/ppmi_matrix.npy` | Dense PPMI matrix, shape `(10000, 10000)` |
| `embeddings/ppmi_matrix_sparse.npz` | Sparse PPMI matrix |
| `models/bilstm_pos.pt` | Fine-tuned BiLSTM POS checkpoint |
| `models/bilstm_pos_frozen.pt` | Frozen-embedding BiLSTM POS checkpoint |
| `models/bilstm_ner.pt` | BiLSTM NER checkpoint |
| `models/transformer_cls.pt` | Transformer classifier checkpoint |

Note: `ppmi_matrix.npy` is large because it stores a dense `10000 x 10000` float32 matrix. The sparse `.npz` version is much smaller and is usually better for storage or submission packaging.

## Dataset Splits

The generated CoNLL files currently contain:

| File | Sentences | Tokens |
| --- | ---: | ---: |
| `data/pos_train.conll` | 350 | 9,173 |
| `data/pos_test.conll` | 75 | 1,989 |
| `data/ner_train.conll` | 350 | 9,173 |
| `data/ner_test.conll` | 75 | 1,989 |

The topic classification dataset is derived from 300 metadata-linked articles using weak category labels.

## Current Results Snapshot

The notebook output includes these final topic-classification results:

| Model | Test Accuracy | Macro-F1 | Best Epoch | Avg Sec/Epoch | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiLSTM Classifier | 0.4222 | 0.3775 | 19 | 0.09 | 1,647,557 |
| Transformer Encoder | 0.4222 | 0.2844 | 18 | 0.27 | 2,080,389 |

The POS fine-tuned model reaches much stronger performance than the frozen-embedding POS variant in the saved notebook outputs. The NER section includes both raw neural decoding and gazetteer-constrained evaluation.

## Notes

- Read Urdu text files with UTF-8 encoding. Some terminals may display mojibake if they default to a legacy Windows encoding.
- The notebook is self-contained and includes explanatory markdown after most major steps.
- The project folder is not currently a git repository.
- If submitting the project, check whether large dense artifacts such as `embeddings/ppmi_matrix.npy` are required, since the sparse version is far smaller.
