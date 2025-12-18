from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json

def load_llm(
    model_name: str,
    load_in_4bit: bool = True,
):
    """
    Load tokenizer and causal LM model.

    Returns:
        tokenizer, model
    """
    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
    )

    return tokenizer, model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
):
    """
    Run the LLM on a single prompt and return decoded generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        do_sample=False,
    )

    gen_ids = outputs[:, input_len:]
    return tokenizer.batch_decode(
        gen_ids,
        skip_special_tokens=True
    )[0]


def safe_parse_llm_json(text: str) -> dict:
    try:
        obj = json.loads(text)
        return {
            "Conclusion": obj.get("Conclusion", "ParseError"),
            "Evidence": obj.get("Evidence", ""),
        }
    except json.JSONDecodeError:
        return {"Conclusion": "ParseError", "Evidence": ""}
