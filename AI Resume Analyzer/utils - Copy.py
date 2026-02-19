from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load pretrained NER model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    entities = ner_pipeline(text)
    result = {}
    for ent in entities:
        label = ent['entity_group']
        if label not in result:
            result[label] = []
        result[label].append(ent['word'])
    # Deduplicate
    for key in result:
        result[key] = list(set(result[key]))
    return result
