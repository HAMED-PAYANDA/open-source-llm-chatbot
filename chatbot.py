# 1. Imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 2. Define the model
model_name = "facebook/blenderbot-400M-distill"

# 3. Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\nChatbot ready! (type 'exit' to quit)\n")

# 4. Initialize the conversation history list
conversation_history = []

# 5. Start the conversation loop
while True:
    # Keep only the last 6 exchanges to prevent overwhelming the model's memory
    conversation_history = conversation_history[-6:]
    
    # Encode the history as a single string with newlines
    history_string = "\n".join(conversation_history)

    # Fetch user input from the terminal
    input_text = input("> ")

    # Check if the user wants to quit
    if input_text.lower() == "exit":
        print("Goodbye!")
        break

    # Format the prompt using the history and new user input
    prompt = history_string + f"\nUser: {input_text}\nBot:"

    # Tokenize the prompt (convert text to numerical tensors)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Generate the bot's response using specific parameters to control creativity and repetition
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )

    # Decode the numerical output back into readable text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Print the bot's response to the terminal
    print("Bot:", response)

    # Save both the user's input and the bot's response to the history list
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

    