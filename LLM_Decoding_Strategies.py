import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
# Using SmolLM2-135M-Instruct (SFT) as requested.
# If "SmolLM2-135M-SFT-Only" refers to a specific private checkpoint, update this ID.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-SFT-Only"
REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2" # Example public reward model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on device: {DEVICE}")

# from datasets import load_dataset
# # Load 50 examples from the Alpaca evaluation set
# ds = load_dataset("tatsu-lab/alpaca", split="train[:50]") 
# prompts = [item['instruction'] for item in ds]

# ==========================================
# 1. DECODING STRATEGY IMPLEMENTATIONS
# ==========================================

def greedy_search(model, tokenizer, prompt, max_new_tokens=50):
    """
    Implements Greedy Search: Selects the token with the highest probability at each step.
    """
    # Encode input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated_ids = input_ids

    # Loop until max length
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :] # Get logits for the last token

            # Apply Softmax (optional for argmax, but good for consistency)
            probs = F.softmax(logits, dim=-1)

            # Argmax to get the best token
            next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def beam_search(model, tokenizer, prompt, max_new_tokens=50, beam_width=3):
    """
    Implements Beam Search: Maintains 'beam_width' most promising sequences.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    # Each candidate is a tuple: (log_probability_score, sequence_tensor)
    # Initialize with the prompt, score 0.0 (log(1) = 0)
    candidates = [(0.0, input_ids)]

    for _ in range(max_new_tokens):
        all_expansions = []

        # Expand each current candidate
        for score, seq in candidates:
            # Check if last token is EOS
            if seq[0, -1].item() == tokenizer.eos_token_id:
                all_expansions.append((score, seq)) # Keep completed sequences
                continue

            with torch.no_grad():
                outputs = model(seq)
                logits = outputs.logits[:, -1, :]
                
                # Use Log Softmax for numerical stability when summing probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # We could expand ALL tokens, but for efficiency, let's just look at top beam_width * 2
                # to reduce computation, or strictly following text: "all possible extensions"
                # For strict implementation of "consider all", we iterate all vocab, 
                # but usually we take top-k to keep it fast. I will do top-k expansion for speed 
                # but conceptual correctness remains.
                top_vals, top_indices = torch.topk(log_probs, k=beam_width * 2)
                
                for i in range(top_vals.shape[1]):
                    next_token_log_prob = top_vals[0, i].item()
                    next_token_id = top_indices[0, i].view(1, 1)
                    
                    new_score = score + next_token_log_prob
                    new_seq = torch.cat([seq, next_token_id], dim=-1)
                    all_expansions.append((new_score, new_seq))

        # Sort by score (highest first) and keep top 'beam_width'
        ordered = sorted(all_expansions, key=lambda x: x[0], reverse=True)
        candidates = ordered[:beam_width]

        # Stop if all top candidates end with EOS (optional heuristic)
        if all(c[1][0, -1].item() == tokenizer.eos_token_id for c in candidates):
            break

    # Return the sequence with the highest score
    best_score, best_seq = candidates[0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)

def top_k_sampling(model, tokenizer, prompt, max_new_tokens=50, k=50, temperature=1.0):
    """
    Implements Top-K Sampling: Samples from the top K tokens after scaling by temperature.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # 1. Scale by Temperature
            logits = logits / temperature

            # 2. Find Top K
            top_k_logits, top_k_indices = torch.topk(logits, k)

            # 3. Create a mask of -inf for all non-top-k tokens
            # (In practice, we just sample from the top_k values directly)
            
            # 4. Softmax on the filtered logits
            probs = F.softmax(top_k_logits, dim=-1)

            # 5. Sample from the distribution
            next_token_index_in_top_k = torch.multinomial(probs, num_samples=1)
            next_token_id = top_k_indices.gather(-1, next_token_index_in_top_k)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def top_p_sampling(model, tokenizer, prompt, max_new_tokens=50, p=0.9, temperature=1.0):
    """
    Implements Top-P (Nucleus) Sampling: Samples from the smallest set of tokens 
    whose cumulative probability exceeds P.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # 1. Scale by Temperature
            logits = logits / temperature

            # 2. Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)

            # 3. Compute cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # 4. Remove tokens with cumulative probability above the threshold
            # Shift the mask right to keep also the first token above threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 5. Mask logits to -inf
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            # 6. Sample
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ==========================================
# 2. METRICS & EVALUATION
# ==========================================

def calculate_distinct_n(texts, n):
    """
    Calculates Distinct-N: unique N-grams / total N-grams.
    Higher values indicate greater diversity.
    """
    if not texts: return 0.0
    
    unique_ngrams = set()
    total_ngrams = 0

    for text in texts:
        words = text.split()
        if len(words) < n:
            continue
        # Generate N-grams
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        unique_ngrams.update(ngrams)
        total_ngrams += len(ngrams)

    if total_ngrams == 0:
        return 0.0
    
    return len(unique_ngrams) / total_ngrams

def get_reward_score(reward_model, tokenizer, prompt, response):
    """
    Uses a pretrained Reward Model to score the quality of the response.
    Returns a scalar score.
    """
    # The reward model expects "Prompt + Response" usually, or specific formatting
    # For many RMs, we can pass text pairs or concatenated text.
    # We will use simple concatenation for this generic implementation.
    inputs = tokenizer(prompt, response, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        score = reward_model(**inputs).logits[0].item()
    return score

# ==========================================
# 3. MAIN EXECUTION (TASK 1 LOGIC)
# ==========================================

def main():
    # Load Models
    print("Loading SFT Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load Reward Model (Quality Metric)
    print("Loading Reward Model...")
    try:
        rm_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
        rm_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    except Exception as e:
        print(f"Error loading reward model: {e}")
        print("Continuing without quality scoring...")
        rm_model = None
        rm_tokenizer = None

    # Define Test Data (Instruction-Following Subset)
    prompts = [
        "Explain quantum entanglement to a 5 year old.",
        "Write a python function to merge two sorted lists.",
        "Give me a recipe for chocolate chip cookies.",
        "What are the main causes of the French Revolution?",
        "Compose a haiku about artificial intelligence."
    ]

    # Parameters
    temperatures = [0.2, 0.5, 0.8, 1.0, 1.2]
    beam_width = 3
    top_k_val = 50
    top_p_val = 0.9
    
    results_log = []

    print("\n--- Starting Analysis ---")

    # 1. BASELINES: GREEDY & BEAM (Deterministic)
    print("\nRunning Deterministic Strategies (Greedy & Beam)...")
    greedy_responses = []
    beam_responses = []
    
    for prompt in prompts:
        # Greedy
        res_g = greedy_search(model, tokenizer, prompt)
        greedy_responses.append(res_g)
        score_g = get_reward_score(rm_model, rm_tokenizer, prompt, res_g) if rm_model else 0
        
        # Beam
        res_b = beam_search(model, tokenizer, prompt, beam_width=beam_width)
        beam_responses.append(res_b)
        score_b = get_reward_score(rm_model, rm_tokenizer, prompt, res_b) if rm_model else 0
        
        results_log.append({"Strategy": "Greedy", "Temp": 0, "Prompt": prompt[:20], "Score": score_g})
        results_log.append({"Strategy": "Beam", "Temp": 0, "Prompt": prompt[:20], "Score": score_b})

    # 2. SAMPLING STRATEGIES (Top-K & Top-P) ACROSS TEMPERATURES
    print("\nRunning Sampling Strategies (Top-K & Top-P)...")
    
    # Storage for Across-Prompt Diversity (at T=0.8 usually, but let's log all)
    # Map: (Strategy, Temp) -> List of responses
    sampling_collections = {} 

    for T in temperatures:
        print(f"Processing Temperature T={T}...")
        
        top_k_res_list = []
        top_p_res_list = []
        
        for prompt in prompts:
            # Top-K
            res_k = top_k_sampling(model, tokenizer, prompt, k=top_k_val, temperature=T)
            score_k = get_reward_score(rm_model, rm_tokenizer, prompt, res_k) if rm_model else 0
            top_k_res_list.append(res_k)
            
            # Top-P
            res_p = top_p_sampling(model, tokenizer, prompt, p=top_p_val, temperature=T)
            score_p = get_reward_score(rm_model, rm_tokenizer, prompt, res_p) if rm_model else 0
            top_p_res_list.append(res_p)
            
            results_log.append({"Strategy": "Top-K", "Temp": T, "Score": score_k})
            results_log.append({"Strategy": "Top-P", "Temp": T, "Score": score_p})

        sampling_collections[("Top-K", T)] = top_k_res_list
        sampling_collections[("Top-P", T)] = top_p_res_list

    # ==========================================
    # 4. REPORTING METRICS
    # ==========================================
    print("\n\n=== RESULTS ===")
    
    # -- Quality (Average Reward Score) --
    print("\n[Quality Analysis (Avg Reward Score)]")
    # Group by Strategy/Temp and average scores
    # (Simple aggregation logic)
    agg_scores = {}
    for entry in results_log:
        key = (entry["Strategy"], entry["Temp"])
        if key not in agg_scores: agg_scores[key] = []
        agg_scores[key].append(entry["Score"])
    
    for (strat, temp), scores in sorted(agg_scores.items()):
        print(f"{strat} (T={temp}): {np.mean(scores):.4f}")

    # -- Diversity (Distinct-N) --
    print("\n[Diversity Analysis (Distinct-2)]")
    
    # Calculate Across-Prompt Diversity for each setting
    # (Using Distinct-2 as representative N-gram metric)
    print("Distinct-2 (Across Prompts):")
    
    # Greedy/Beam
    d2_greedy = calculate_distinct_n(greedy_responses, 2)
    d2_beam = calculate_distinct_n(beam_responses, 2)
    print(f"Greedy: {d2_greedy:.4f}")
    print(f"Beam:   {d2_beam:.4f}")
    
    # Sampling
    for (strat, temp), responses in sorted(sampling_collections.items()):
        d2 = calculate_distinct_n(responses, 2)
        print(f"{strat} (T={temp}): {d2:.4f}")

    # -- Within-Prompt Diversity Analysis (at Fixed T=0.8) --
    print("\n[Within-Prompt Diversity (T=0.8, Prompt 1)]")
    fixed_prompt = prompts[0]
    n_samples = 5
    
    # Generate N samples for single prompt
    print(f"Generating {n_samples} samples for prompt: '{fixed_prompt}'...")
    
    samples_k = [top_k_sampling(model, tokenizer, fixed_prompt, k=top_k_val, temperature=0.8) for _ in range(n_samples)]
    samples_p = [top_p_sampling(model, tokenizer, fixed_prompt, p=top_p_val, temperature=0.8) for _ in range(n_samples)]
    
    wd_k = calculate_distinct_n(samples_k, 2)
    wd_p = calculate_distinct_n(samples_p, 2)
    
    print(f"Top-K (T=0.8) Within-Prompt Distinct-2: {wd_k:.4f}")
    print(f"Top-P (T=0.8) Within-Prompt Distinct-2: {wd_p:.4f}")

if __name__ == "__main__":
    main()
