from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load pretrained NER model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    entities = []
    # Chunk text to avoid max sequence length issues (limit is usually 512 tokens)
    # Using a safe character limit (approx 300 words or 2000 chars might be safe for 512 tokens, conservative 1000 chars)
    chunk_size = 1000 
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        try:
            chunk_entities = ner_pipeline(chunk)
            entities.extend(chunk_entities)
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

    result = {}
    for ent in entities:
        label = ent['entity_group']
        if label not in result:
            result[label] = []
        result[label].append(ent['word'])
    
    # Deduplicate and sort
    for key in result:
        result[key] = sorted(list(set(result[key])))
    return result
