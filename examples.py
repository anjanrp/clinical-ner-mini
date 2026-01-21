import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "tner/roberta-large-bc5cdr"

device = "mps" if torch.backends.mps.is_available() else -1

tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
mdl = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "token-classification",
    model=mdl,
    tokenizer=tok,
    aggregation_strategy="simple",
    device=device,
)

texts = [
    "Patient was started on aspirin and metformin for type 2 diabetes.",
    "The subject developed pneumonia and was treated with azithromycin.",
    "Warfarin was discontinued due to gastrointestinal bleeding.",
    "Exposure to benzene is associated with hematologic malignancy.",
    "No evidence of myocardial infarction was found on evaluation.",
]

for t in texts:
    print("\nTEXT:", t)
    for ent in ner(t):
        print(f"  - {ent['word']} | {ent['entity_group']} | score={ent['score']:.3f}")
