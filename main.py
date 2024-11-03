import torch
import gc
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM


def read_from_file(file_path: str):
    sentences = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            sentences.append(line.strip())
    return sentences


def load_llm(model_name: str, device: torch.device):
    match model_name:
        case name if name.startswith("Llama"):
            model = LlamaForCausalLM.from_pretrained(model_name, use_safetensors=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
            return model, tokenizer
        case name if name.startswith("Qra"):
            model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
            return model, tokenizer
        case _:
            raise ValueError("Invalid model_name.")


def calculate_perplexity(model, tokenizer, prompts: list[str]) -> float:
    perplexity_sum = 0.0
    for prompt in prompts:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # The loss is the mean cross-entropy loss over the sequence
            # in PyTorch's implementation of transformers.
            loss = outputs.loss

        # Perplexity is the exponent of the loss
        perplexity = torch.exp(loss)
        perplexity_sum += perplexity.item()
    return perplexity_sum / len(prompts)


def calculate_perplexity_qra(model_id: str, prompts: list[str], device: torch.device) -> float:
    model = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_safetensors=True)

    perplexity_sum = 0.0
    for prompt in prompts:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # The loss is the mean cross-entropy loss over the sequence
            # in PyTorch's implementation of transformers.
            loss = outputs.loss

        # Perplexity is the exponent of the loss
        perplexity = torch.exp(loss)
        perplexity_sum += perplexity.item()

    return perplexity_sum / len(prompts)


if __name__ == "__main__":
    sentences = read_from_file(file_path="/home/ola/Downloads/pl.txt")[:10]

    model, tokenizer = load_llm("Llama-3.2-1B", torch.device(0))
    avg_perplexity_llama_1b = calculate_perplexity(model, tokenizer, prompts=sentences)
    print(f"{avg_perplexity_llama_1b=}")
    model.to('cpu')  # remove model from GPU
    torch.cuda.empty_cache()

    model, tokenizer = load_llm("Llama-3.2-3B", torch.device("cpu"))
    avg_perplexity_llama_3b = calculate_perplexity(model, tokenizer, prompts=sentences)
    print(f"{avg_perplexity_llama_3b=}")

    model, tokenizer = load_llm("Qra-1b", torch.device(0))
    avg_perplexity_qra_1b = calculate_perplexity(model, tokenizer, prompts=sentences)
    print(f"{avg_perplexity_qra_1b=}")
