from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import re

def load_llm(
    model_name: str,
    load_in_4bit: bool = True,
):
    """
    Load tokenizer and LM model.

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.eval()

    return tokenizer, model


# def generate_text(
#     model,
#     tokenizer,
#     prompt: str,
#     max_new_tokens: int = 96,
#     max_input_tokens: int = 7000,
#     min_new_tokens: int = 0,
# ):
#     """
#     Run the LLM on a single prompt and return decoded generated text.
#     Limits prompt size and keeps the end of the prompt.
#     """

#     # --- NEW: apply chat template if the model expects it ---
#     if tokenizer.chat_template is not None:
#         messages = [{"role": "user", "content": prompt}]
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#     # Make truncation keep the *end* of the prompt (often where instructions sit)
#     tokenizer.truncation_side = "left"

#     # Ensure pad token is defined (Mistral often has no pad_token by default)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # Tokenise with truncation to control prompt size
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_input_tokens,
#     ).to(model.device)

#     ids = inputs.input_ids[0]
#     print("HEAD preview:\n", tokenizer.decode(ids[:250], skip_special_tokens=False))
#     print("TAIL preview:\n", tokenizer.decode(ids[-300:], skip_special_tokens=False))

#     input_len = inputs.input_ids.shape[1]

#     print(
#         f"Input tokens (after trunc): {input_len} | "
#         f"Max new tokens: {max_new_tokens} | "
#         f"Total (worst case): {input_len + max_new_tokens}",
#         flush=True
#     )

#     with torch.inference_mode():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             min_new_tokens=min_new_tokens,
#             num_beams=1,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     out_len = outputs.shape[1]
#     new_len = out_len - input_len
#     print(f"Generated new tokens: {new_len}", flush=True)

#     gen_ids = outputs[:, input_len:]
#     text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

#     # Helpful when output is only whitespace
#     if not text.strip():
#         print("Warning: generation returned empty/whitespace output", flush=True)

#     return text


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    max_input_tokens: int = 2048
):
    
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    tokenizer.truncation_side = "left"  # keeps the end of the prompt

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]
    ids = inputs.input_ids[0]
    print("HEAD preview:\n", tokenizer.decode(ids[:200], skip_special_tokens=False))
    print("TAIL preview:\n", tokenizer.decode(ids[-300:], skip_special_tokens=False))   

    print(
        f"Input tokens (after trunc): {input_len} | "
        f"Max new tokens: {max_new_tokens} | "
        f"Total (worst case): {input_len + max_new_tokens}",
        flush=True
    )



    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[:, input_len:]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


def safe_parse_llm_json(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not m:
        return {"Conclusion": "ParseError", "Evidence": ""}

    try:
        obj = json.loads(m.group(0))
        return {
            "Conclusion": obj.get("Conclusion", "ParseError"),
            "Evidence": obj.get("Evidence", ""),
        }
    except json.JSONDecodeError:
        return {"Conclusion": "ParseError", "Evidence": ""}
