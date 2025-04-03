from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def preprocess(data):

    FLAN_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    FLAN_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    # Combining questions and options into a single string
    inputs = [
        f"Question: {q} Options: A. {a}, B. {b}, C. {c}, D. {d} Answer:"
        for q, a, b, c, d in zip(
            data["question"], data["opa"], data["opb"], 
            data["opc"], data["opd"])
    ]

    # The target is the correct option(s)
    targets = data["cop"]

    # Tokenize inputs
    model_inputs = FLAN_tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = FLAN_tokenizer(targets, max_length=48, truncation=True, padding="max_length")

    # Replacing the padding token id with -100 to ignore it in loss calculation
    labels["input_ids"] = [
        [(l if l != FLAN_tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs