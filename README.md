# Clinical/Biomedical NER Mini-Project (Pretrained Transformer Evaluation)

This mini-project demonstrates an **off-the-shelf transformer NER** workflow for extracting **structured entities** from **unstructured biomedical text** and reporting standard precision/recall/F1 metrics—similar to how clinical NLP pipelines create “NLP-enriched” variables for downstream analytics.

---

## Goal
Evaluate a **pretrained biomedical Named Entity Recognition (NER)** model (no training) that labels:
- **Chemical**
- **Disease**

…and report metrics + qualitative examples.

---

## Dataset + Model
- **Dataset:** `tner/bc5cdr` (entity types: Chemical, Disease)
- **Model:** `tner/roberta-large-bc5cdr` (transformer token classification)
- **Metric:** `seqeval` (precision/recall/F1 for sequence labeling)

---

## Environment Setup
```bash
conda create -n clinical-ner python=3.11 -y
conda activate clinical-ner
pip install torch transformers datasets evaluate seqeval accelerate
```

## How to Run

### 1) Evaluate on test split
```bash
python evaluate_ner.py --split test --batch_size 4 --model tner/roberta-large-bc5cdr
```

### 2) Generate example extractions
```bash
python examples.py
```

## Results (Test Split)

**Device:** `mps` (Apple Silicon GPU via PyTorch MPS)

### Entity-level

| Entity Type | Precision | Recall | F1 | # Entities |
|---|---:|---:|---:|---:|
| Chemical | 0.9130 | 0.9354 | 0.9241 | 5385 |
| Disease | 0.7935 | 0.8486 | 0.8201 | 4424 |

### Overall
- **Overall Precision:** 0.8578  
- **Overall Recall:** 0.8962  
- **Overall F1:** 0.8766  
- **Token Accuracy:** 0.9761  

---


## Example Extractions (Qualitative)

**TEXT:** Patient was started on aspirin and metformin for type 2 diabetes.
~~~text
aspirin | Chemical | score=1.000
metformin | Chemical | score=1.000
type 2 diabetes | Disease | score=1.000
~~~

**TEXT:** The subject developed pneumonia and was treated with azithromycin.
~~~text
pneumonia | Disease | score=1.000
azithromycin | Chemical | score=1.000
~~~

**TEXT:** Warfarin was discontinued due to gastrointestinal bleeding.
~~~text
Warfarin | Chemical | score=0.998
gastrointestinal bleeding | Disease | score=1.000
~~~

**TEXT:** Exposure to benzene is associated with hematologic malignancy.
~~~text
benzene | Chemical | score=1.000
hematologic malignancy | Disease | score=0.996
~~~

**TEXT:** No evidence of myocardial infarction was found on evaluation.
~~~text
myocardial infarction | Disease | score=1.000
~~~

---

## Notes / Limitations
- The model extracts entities but does **not** capture clinical assertion status (e.g., **negation**).  
  Example: “No evidence of myocardial infarction” still contains the Disease entity span.
- Next steps for a clinical pipeline would include **negation detection**, **section-aware parsing**, and **normalization** (mapping extracted mentions to controlled vocabularies).

