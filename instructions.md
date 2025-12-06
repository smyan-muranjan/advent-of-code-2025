**Role:** You are an expert AI Engineer specializing in Reinforcement Learning.
**Task:** Implement a "Tiny GRPO" (Group Relative Policy Optimization) fine-tuning pipeline from scratch to train a small Language Model on a toy task.
**Output Format:** Do **not** create a Jupyter Notebook. Create a **single Python script** (`.py`). You must use clear comment separators (e.g., `### CELL 1: ...`) to denote the distinct sections required by the assignment structure.

## Project Requirements
You must use a tiny decoder-only model (e.g., `GPT-Neo-125M` or `TinyLlama`) to ensure it runs on limited hardware. You will implement the GRPO algorithm manually (no `trl` or `ppo` libraries for the loss calculation).

### 1. The Toy Task
Select a deterministic, auto-evaluable toy task.
* **Recommendation:** "Count the specific character."
* **Prompt structure:** "Count the number of 'a's in: [random string]. Output:"
* **Target:** The model should output the single integer representing the count.

### 2. Implementation Structure (The Python File)
Organize your code using the following "Cell" markers as comments:

**### CELL 0: Imports & Config**
* Import `torch`, `transformers` (AutoTokenizer, AutoModelForCausalLM), and standard libraries.
* Set device (CUDA/MPS/CPU).
* Set hyperparameters (Group Size $G \ge 2$, Learning Rate, etc.).

**### CELL 1: Model Loading**
* Load the pretrained model and tokenizer (e.g., `EleutherAI/gpt-neo-125m`).
* Ensure the model is trainable.

**### CELL 2: Dataset Generation**
* Create a class or function to generate the dataset.
* Generate 500+ training examples and 50+ evaluation examples.
* **Action:** Print 3 example prompts from the dataset.

**### CELL 3: Reward Function**
* Implement `get_reward(prompts, responses)`.
* Logic: Parse the integer from the model response. Compare it to the ground truth count.
    * Return `1.0` for correct.
    * Return `0.0` for incorrect.
    * (Optional) Return partial credit (e.g., `0.1`) if the format is correct but the number is wrong.
* **Action:** Perform a "Sanity Check." Create a manual list of prompts and known good/bad responses. Run them through the function and print the rewards to verify accuracy.

**### CELL 4: GRPO Loss Implementation**
* Implement the core GRPO logic. This is the most critical section.
* **Algorithm:**
    1.  **Sample:** For a given prompt, generate a group of $G$ completions (where $G \ge 2$).
    2.  **Score:** Calculate rewards $r$ for all completions in the group.
    3.  **Advantage:** Compute normalized advantages ($A_i$) for each completion using the group mean and std:
        $$A_{i}=\frac{r_{i}-\text{mean}(\{r_{1}...r_{G}\})}{\text{std}(\{r_{1}...r_{G}\})+\epsilon}$$
    4.  **Loss:** Compute the policy gradient loss:
        $$\text{Loss} = -\frac{1}{G}\sum_{i=1}^{G}[A_{i} \cdot \log \pi_{\theta}(y_{i}|x)]$$
        *(Note: The negative sign is because we want to maximize reward using gradient descent).*
    5.  **KL (Optional):** You may include a simple KL divergence approximation if feasible, but the core requirement is the advantage-based loss.

**### CELL 5: Training Loop**
* Initialize `AdamW` optimizer.
* Loop through the dataset.
* For each step:
    * Generate the group of outputs.
    * Calculate Loss.
    * Backpropagate and Step.
* **Logging:** Print the Average Reward and Average Loss every 10 steps.

**### CELL 6: Evaluation & Generation**
* **Quantitative:** Run the fine-tuned model on the held-out Evaluation Set. Calculate mean reward.
* **Qualitative:** Generate responses for 5 specific prompts.
* **Output:** Print a comparison: "Baseline Reward vs. Fine-tuned Reward". Print the 5 generated examples.

**### CELL 7: Analysis**
* Print a multi-line string containing a brief analysis answering:
    1.  Did the model learn the task?
    2.  Did you observe reward hacking?
    3.  How does Group Size ($G$) impact stability?

## Technical Constraints
* **Group Size:** You must sample at least 2 outputs per prompt ($G \ge 2$).
* **Tokenization:** Handle padding correctly (left-padding is usually required for generation batches).
* **Gradient Checkpointing:** Enable if needed for memory, though a 125M model should fit easily.

Write the complete code now.