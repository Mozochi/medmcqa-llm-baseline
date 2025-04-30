from transformers import PreTrainedTokenizer 



def preprocess(data, tokenizer: PreTrainedTokenizer): 
    # Preprocesses data for FLAN-T5 using a provided tokenizer

    FLAN_tokenizer = tokenizer # Using the passed tokenizer

    inputs = [
        f"Question: {q} Options: A. {a}, B. {b}, C. {c}, D. {d} Answer:"
        for q, a, b, c, d in zip(
            data["question"], data["opa"], data["opb"],
            data["opc"], data["opd"])
    ]

    targets = [str(item) for item in data["cop"]]
    num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    targets = [num_to_letter.get(int(t), str(t)) for t in targets]

    model_inputs = FLAN_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = FLAN_tokenizer(targets, max_length=8, truncation=True, padding="max_length")

    labels_input_ids = labels["input_ids"]
    processed_labels = [
        [(l if l != FLAN_tokenizer.pad_token_id else -100) for l in label]
        for label in labels_input_ids
    ]

    model_inputs["labels"] = processed_labels
    return model_inputs