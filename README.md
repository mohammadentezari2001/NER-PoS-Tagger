## NLP Sequence Tagging: PoS and Named Entity Recognition

This repository contains two primary modules demonstrating fundamental and advanced techniques for Natural Language Processing (NLP) sequence tagging tasks: **Part-of-Speech (PoS) Tagging** and **Named Entity Recognition (NER)**.

The project explores both a simple statistical baseline (Most Frequent Tagger) and a deep learning approach (Bi-directional GRU) for PoS tagging, alongside a production-ready custom training pipeline using spaCy for NER.

### Features

*   **Part-of-Speech (PoS) Tagging:**
    *   **Statistical Baseline:** Implementation of a simple Most Frequent Tagger for a strong baseline performance.
    *   **Deep Learning Model:** A custom Bi-directional GRU (Gated Recurrent Unit) model built with PyTorch for enhanced sequence prediction.
*   **Named Entity Recognition (NER):**
    *   Custom training pipeline using the powerful **spaCy** library.
    *   Robust data preprocessing and entity alignment logic to handle common issues like overlapping and misaligned entities in noisy JSONL data.
    *   Detailed evaluation metrics at both the **Entity Level** and **Token Level**.
*   **Technologies:** PyTorch, spaCy, scikit-learn.


### Datasets

This project utilizes publicly available datasets tailored for the specific sequence tagging tasks:

#### 1. Named Entity Recognition (NER) Dataset

*   **Name:** Resume Entities for NER
*   **Source:** [Kaggle Dataset by DataTurks](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner)
*   **Content:** This dataset consists of **220 labeled resumes** intended for training NER models for resume parsing. The data is in JSONL format, where each entry contains the resume text (`content`) and a list of entities (`annotation`) marked by character offsets.
*   **Entity Categories (10 Labels):**
    *   `Name`
    *   `College Name`
    *   `Degree`
    *   `Graduation Year`
    *   `Years of Experience`
    *   `Companies worked at`
    *   `Designation`
    *   `Skills`
    *   `Location`
    *   `Email Address`
*   **Note on Usage:** The source data is known to contain issues such as overlapping or non-token-aligned entity spans, which are typical challenges when working with crowdsourced or automatically annotated data. The NER script includes robust preprocessing logic to filter and align these entities using spaCy's utility functions (`offsets_to_biluo_tags`) to ensure successful model training.

#### 2. Part-of-Speech (PoS) Tagging Dataset

*   **Name:** UD\_English-GUM (Universal Dependencies - Georgetown University Multilayer corpus)
*   **Source:** [Universal Dependencies GitHub repository (UD\_English-GUM)](https://github.com/UniversalDependencies/UD_English-GUM/tree/5ae58e9b57c907e6047cad319beff9dce940f391)
*   **Content:** This is an open-source, richly annotated corpus of English texts from multiple genres (e.g., academic, non-fiction, travel guides). It is a part of the Universal Dependencies project.
*   **Annotation:** The data is manually corrected to include **Universal Part-of-Speech (UPOS)** tags, along with morphological features and dependency relations. The corpus contains a large number of annotated tokens (over 229,000 tokens in the referenced version), making it a comprehensive resource for training and evaluating PoS taggers.
*   **File Format:** The notebook uses a pre-processed version of this dataset, likely in a JSONL format (`train.json`, `test.json`), where each entry contains `words` and corresponding `labels` (PoS tags).


### Project Structure

The core logic is contained within two Jupyter Notebooks:

| File Name | Description |
| :--- | :--- |
| `PoS_Tagger.ipynb` | Contains the implementation for the Most Frequent Tagger and the Bi-directional GRU PoS Tagger on the provided dataset. |
| `NER.ipynb` | Contains the custom spaCy pipeline for training a Named Entity Recognition model on a resume-based dataset. |

### Setup and Installation

The following dependencies are required to run the notebooks.

```bash
# General NLP/DL Dependencies
!pip install torch numpy scikit-learn tqdm nlp

# Force reinstall PyTorch components (as done in the notebook)
# Requires a session restart after the first install.
!pip install --upgrade --force-reinstall torch torchvision torchaudio

# SpaCy and Table Dependencies for NER
!pip install spacy tabulate
```

### Key Results

#### Part-of-Speech (PoS) Tagging Comparison

The GRU-based deep learning model significantly outperformed the statistical baseline on the test set.

| Model | Accuracy (Token Level) | Notes |
| :--- | :--- | :--- |
| **Most Frequent Tagger** (Baseline) | **0.9178** (91.8%) | Simple, fast, excellent for frequent words. |
| **Bi-directional GRU Model** | **0.9359** (93.6%) | Learns contextual information, providing higher accuracy. |

#### Named Entity Recognition (NER) Metrics

The spaCy NER model was trained on custom resume data. The results below show the performance on the validation set, broken down by entity type at both the token and NER span level.

**Token Level Metrics (Overall Weighted Average)**

| Metric | Score |
| :--- | :--- |
| **Precision** | 0.924 |
| **Recall** | 0.915 |

**Entity Level F1-Scores (Test Set)**

| Label | F1-Score | Support |
| :--- | :--- | :--- |
| `Name` | **0.837** | 22 |
| `Degree` | **0.700** | 28 |
| `Email Address` | **0.684** | 15 |
| `College Name` | 0.571 | 28 |
| `Designation` | 0.489 | 56 |
| `Location` | 0.431 | 38 |
| `Companies worked at` | 0.298 | 52 |
| `Skills` | 0.148 | 38 |
| `Years of Experience` | 0.000 | 4 |

***Note on NER Metrics:*** *The low F1-scores for some labels (`Skills`, `Years of Experience`, `Companies worked at`) and the warnings about entity misalignment indicate the dataset is small, noisy, and challenging for token boundary detection, typical of real-world resume parsing tasks.*
