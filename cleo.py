from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define device
device = "cuda"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16,  # Use float16 for faster GPU processing
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

print("C.L.E.O Active, Please state your message! Type 'exit' or 'quit' to end the chat.")

# Interactive loop
while True:
    # User input
    prompt = input("\nYou: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting... Goodbye!")
        break

    # Prepare messages
    system_message = "C.L.E.O Active, Please state your message"
    input_text = f"{system_message}\nUser: {prompt}\nAssistant:"

    # Tokenize input and add attention mask
    model_inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Explicitly set attention_mask
    if tokenizer.pad_token_id is not None:
        model_inputs["attention_mask"] = (model_inputs.input_ids != tokenizer.pad_token_id).long()

    # Generate response
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        temperature=1.0
    )

    # Decode and display response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Remove the system/user labels if present
    response_cleaned = response.replace(system_message, "").replace(f"User: {prompt}", "").strip()
    print(f"\nChatbot: {response_cleaned}")
