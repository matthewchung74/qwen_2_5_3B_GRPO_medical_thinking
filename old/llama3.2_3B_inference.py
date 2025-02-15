from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)


# Inference code
from unsloth import FastLanguageModel
from vllm import SamplingParams
import os

max_seq_length = 512

base_model_name = "Qwen/Qwen2.5-3B-Instruct"
base_model_name = "meta-llama/Llama-3.2-3B"
user_name = "matthewchung74"
model_name = "Qwen2.5_3B-GRPO-medical-reasoning"
model_name = "Llama3.2_3B-GRPO-medical-reasoning"

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"{user_name}/{model_name}",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank = lora_rank,
    token=os.environ["HF_TOKEN"],
    gpu_memory_utilization = 0.3, # Reduce if out of memory
)

# Load LoRA weights
# lora_request = model.load_lora(f"{user_name}/{model_name}")

# Prepare model for inference
model = FastLanguageModel.for_inference(model)

def generate_response(prompt):
    input_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
    )
    
    output = model.fast_generate(
        input_text,
        sampling_params=sampling_params,
    )
    return output[0].outputs[0].text.strip()


# Evaluation code
import evaluate
from tqdm import tqdm
import numpy as np

# Initialize metrics
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

# Convert to pandas and select samples for evaluation
# eval_subset = eval_dataset.to_pandas().head(3)  # Using 3 samples for debugging
eval_subset = eval_dataset.to_pandas().head(100)

predictions = []
references = []

# Generate predictions
for _, sample in tqdm(eval_subset.iterrows(), total=len(eval_subset), desc="Evaluating"):
    prediction = generate_response(sample["prompt"])
    predictions.append(prediction)
    references.append(sample["answer"].strip())

# Calculate metrics
rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
bleu_score = bleu_metric.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

# Calculate average BERTScore F1
bertscore_f1 = np.mean(bertscore_results["f1"])

print("\n--- Evaluation Results ---")
print(f"ROUGE Scores: {rouge_scores}")
print(f"BLEU Score: {bleu_score['bleu']}")
print(f"BERTScore F1: {bertscore_f1:.4f}")
