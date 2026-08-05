# 1. Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# Suppress Hugging Face warnings for a cleaner terminal
warnings.filterwarnings("ignore")

# 2. Define the modern causal LLM
model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading model and tokenizer...")

# 3. Load Tokenizer and set padding token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token

# 4. Load the Model (optimized for CPU and memory)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map="cpu",
  torch_dtype=torch.float32
)

# 5. Initialize the structured message history with a System Prompt
messages = [
  {
      "role": "system",
      "content": "You are a helpful AI assistant. Give short and concise answers in 2-3 lines."
  }
]

print("\nChatbot started. Type 'exit' to quit.\n")

# 6. Start the conversation loop
while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add the user's message to the history
    messages.append({"role": "user", "content": user_input})

    # Keep the system prompt (index 0) + only the last 10 exchanges to save memory
    messages = [messages[0]] + messages[-10:]

    # Apply the modern chat template to format the prompt correctly for the model
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=512
    )

    # Generate response in inference mode (saves memory/compute since we aren't training)
    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=60,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the output, slicing it to ignore the input prompt and only show the new reply
    response = tokenizer.decode(
        outputs[0][tokenized["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print(f"Bot: {response}\n")

    # Save the assistant's response to the history
    messages.append({"role": "assistant", "content": response})
    