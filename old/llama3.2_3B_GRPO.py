#!/usr/bin/env python
import os
import time
import torch
from dotenv import load_dotenv
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from trl import GRPOConfig, GRPOTrainer

# ---------------------------
# Global Chat Template String
# ---------------------------
DEFAULT_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
{{ messages[0]['content'] }}
<</SYS>>

{% for message in messages[1:] %}
{% if message['role'] == 'user' %}
{{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}
{{ message['content'] }} </s><s>[INST]
{% endif %}
{% endfor %}"""

# Global variable for the perplexity calculator
perplexity_calculator = None

# ---------------------------
# Environment & Setup Functions
# ---------------------------
def setup_environment():
    """Load environment variables and configure unsloth temporary directory."""
    load_dotenv()
    unsloth_dir = os.path.join(os.getcwd(), "unsloth_files")
    os.makedirs(unsloth_dir, exist_ok=True)
    os.environ["UNSLOTH_TEMP_DIR"] = unsloth_dir
    # Patch fast RL for GRPO
    PatchFastRL("GRPO", FastLanguageModel)
    # Set HuggingFace cache directory
    os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache/huggingface')

# ---------------------------
# Dataset Functions
# ---------------------------
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def get_medical_questions(split="train") -> Dataset:
    """Load and process the medical questions dataset."""
    data = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en')[split]
    df = data.to_pandas()

    if split == "train":
        train_df, temp_df = train_test_split(df, test_size=0.01, random_state=42)
        eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        test_dataset = Dataset.from_pandas(test_df)

        def map_fn(x):
            return {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['Question']}
                ],
                'answer': x['Response'],    # Reference answer
                'Response': x['Response']   # Also include raw response for simulation
            }

        train_dataset = train_dataset.map(map_fn)
        eval_dataset = eval_dataset.map(map_fn)
        test_dataset = test_dataset.map(map_fn)

        return train_dataset, eval_dataset, test_dataset
    else:
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['Question']}
            ],
            'answer': x['Response'],
            'Response': x['Response']
        })
        return data

def test_reward_function(test_dataset):
    """Test the combined reward function on the provided dataset."""
    test_prompts = []
    test_completions = []
    test_answers = []
    for example in test_dataset:
        test_prompts.append(example["prompt"])
        test_completions.append([{"content": example["Response"]}])
        test_answers.append(example["answer"])

    start_time = time.time()
    rewards = combined_reward_func(test_prompts, test_completions, test_answers)
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(test_prompts)} samples in {elapsed:.2f}s ({elapsed/len(test_prompts):.2f}s per sample)")
    print("\nTraining Data Test Results:")
    for i, (prompt, completion, reward) in enumerate(zip(test_prompts, test_completions, rewards)):
        print(f"\nSample {i+1}:")
        print("Prompt:", prompt[-1]["content"])
        print("Generated:", completion[0]["content"][:100] + "...")
        print("Reference Answer:", test_answers[i][:100] + "...")
        print(f"Reward: {reward:.4f}")

# ---------------------------
# Model, Tokenizer, and PEFT Setup Functions
# ---------------------------
def load_model_and_tokenizer(base_model_name: str, max_seq_length: int, lora_rank: int):
    """Load the model and tokenizer and apply PEFT modifications."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,         # Use 4-bit loading; set False for LoRA 16-bit
        fast_inference=True,       # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5, # Adjust if out-of-memory errors occur
    )
    # Set chat template to avoid templating errors later
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set the correct pad token for text-only models
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Verify that necessary tokens exist
    assert tokenizer.pad_token in tokenizer.get_vocab(), "Pad token missing!"
    assert tokenizer.eos_token in tokenizer.get_vocab(), "EOS token missing!"
    print("Tokenizer vocab size:", len(tokenizer))
    print("Max token ID:", max(tokenizer.get_vocab().values()))
    return model, tokenizer

def init_perplexity_calculator():
    """Instantiate the global PerplexityCalculator."""
    global perplexity_calculator
    perplexity_calculator = PerplexityCalculator()

# ---------------------------
# Perplexity Calculator & Reward Functions
# ---------------------------
class PerplexityCalculator:
    def __init__(self, model_name="microsoft/biogpt", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def calculate(self, texts: List[str], batch_size=8) -> List[float]:
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if not batch:
                continue
            encodings = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss
            if torch.isnan(loss):
                raise ValueError("NaN loss encountered")
            batch_perplexity = torch.exp(loss).repeat(len(batch)).cpu().tolist()
            perplexities.extend(batch_perplexity)
        return perplexities

def semantic_correctness(responses: List[str], answers: List[str],
                         device="cuda" if torch.cuda.is_available() else "cpu") -> List[float]:
    """Calculate semantic similarity using a cross-encoder."""
    model_ce = CrossEncoder('cross-encoder/stsb-roberta-base', device=device)
    with torch.no_grad():
        inputs = list(zip(responses, answers))
        return model_ce.predict(inputs, show_progress_bar=False).tolist()

def combined_reward_func(
    prompts: List[List[Dict[str, str]]],
    completions: List[List[Dict[str, str]]],
    answers: List[str],
    **kwargs,
) -> List[float]:
    """Combine semantic similarity and perplexity rewards."""
    global perplexity_calculator
    responses = []
    valid_indices = []
    for idx, completion in enumerate(completions):
        try:
            response = completion[0]['content'].strip()
            if not response:
                continue
            responses.append(response)
            valid_indices.append(idx)
        except (KeyError, IndexError):
            continue

    if not responses:
        return [-1.0] * len(completions)

    try:
        similarities = semantic_correctness(responses, [answers[i] for i in valid_indices])
        perplexities = perplexity_calculator.calculate(responses)
    except Exception as e:
        print(f"Reward calculation error: {str(e)}")
        return [-1.0] * len(completions)

    sim_scores = torch.nan_to_num(torch.tensor(similarities), nan=0.0)
    perplex_scores = torch.nan_to_num(torch.tensor(perplexities), nan=1000.0)

    perplex_rewards = 1 / (perplex_scores / (perplex_scores.mean() + 1e-9))
    score_range = perplex_rewards.max() - perplex_rewards.min()
    if score_range < 1e-6:
        perplex_rewards_normalized = torch.ones_like(perplex_rewards) * 0.5
    else:
        perplex_rewards_normalized = (perplex_rewards - perplex_rewards.min()) / score_range

    combined = [
        0.6 * sim.item() + 0.4 * pr.item()
        for sim, pr in zip(sim_scores, perplex_rewards_normalized)
        if not torch.isnan(sim) and not torch.isnan(pr)
    ]
    final_rewards = [-1.0] * len(completions)
    for idx, reward in zip(valid_indices, combined):
        final_rewards[idx] = max(min(reward, 1.0), -1.0)
    assert len(final_rewards) == len(completions), "Reward mapping error"
    return final_rewards

# ---------------------------
# Training & Inference Functions
# ---------------------------
def get_training_args(checkpoint_repo: str, train_dataset, per_device_train_batch_size: int,
                      gradient_accumulation_steps: int, num_checkpoints: int = 10) -> GRPOConfig:
    """Generate training arguments for GRPO."""
    total_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=6,
        max_prompt_length=256,
        max_completion_length=300,
        max_steps=total_steps,
        save_steps=total_steps // num_checkpoints,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=os.path.join(os.path.expanduser('~'), '.cache/huggingface', checkpoint_repo),
        save_strategy="steps",
        hub_model_id=checkpoint_repo,
        push_to_hub=True,
    )
    return training_args

def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args) -> GRPOTrainer:
    """Create and return a GRPOTrainer instance."""
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer

def run_inference(model, tokenizer, eval_dataset):
    """Run inference on the first evaluation sample and print results."""
    first_test_sample = eval_dataset[0]
    # Ensure the chat template is set before applying it
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    text = tokenizer.apply_chat_template(
        first_test_sample['prompt'],
        tokenize=False,
        add_generation_prompt=True
    )
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=1024,
    )
    # Save and load LoRA weights for generation
    model.save_lora("grpo_saved_lora")
    output = model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text
    user_question = first_test_sample['prompt'][1]['content']
    print("\n===== User Question (Ground Truth) =====")
    print(user_question)
    print("\n===== Reference Answer (Ground Truth) =====")
    print(first_test_sample['answer'])
    print("\n===== Model Output (Generated Response) =====")
    print(output)

def save_and_push_model(model, tokenizer, user_name: str, model_name_for_push: str):
    """Save the merged model and push it to the HuggingFace Hub if HF_TOKEN is set."""
    model.save_pretrained_merged(model_name_for_push, tokenizer, save_method="merged_16bit")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        model.push_to_hub_merged(f"{user_name}/{model_name_for_push}", tokenizer, save_method="merged_16bit", token=hf_token)
    else:
        print("HF_TOKEN not set; skipping push to hub.")

# ---------------------------
# Main Function
# ---------------------------
def main():
    setup_environment()

    # Load datasets
    print("Loading datasets...")
    train_dataset, eval_dataset, test_dataset = get_medical_questions(split="train")
    print("Train dataset sample keys:", train_dataset[0].keys())
    print("Sample prompt:", train_dataset[0]['prompt'])

    # Model configuration
    max_seq_length = 1024
    lora_rank = 64
    base_model_name = "meta-llama/Llama-3.2-3B"
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_name, max_seq_length, lora_rank)

    # Initialize reward-related components
    init_perplexity_calculator()
    print("Testing reward function on test samples...")
    test_reward_function(test_dataset)

    # Training setup
    checkpoint_repo = "matthewchung74/Llama3.2_3B-GRPO-medical-reasoning-checkpoints"
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    training_args = get_training_args(checkpoint_repo, train_dataset, per_device_train_batch_size, gradient_accumulation_steps)
    trainer = create_trainer(model, tokenizer, train_dataset, test_dataset, training_args)

    print("Starting training...")
    trainer.train()

    # Inference
    run_inference(model, tokenizer, eval_dataset)

    # Save and push the model
    user_name = "matthewchung74"
    model_name_for_push = "Llama3.2_3B-GRPO-medical-reasoning"
    save_and_push_model(model, tokenizer, user_name, model_name_for_push)

if __name__ == "__main__":
    main()
