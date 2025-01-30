from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3  # Text-to-Speech Library
import speech_recognition as sr  # Speech-to-Text Library
import re  # For response cleaning

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)  # Adjust speed if needed
tts_engine.setProperty("volume", 1.0)  # Set volume (0.0 to 1.0)

# Initialize STT engine
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

# Announce system activation
boot_message = "C.L.E.O Active, Please state your message! Say 'exit' or 'quit' to end the chat."
print(boot_message)
tts_engine.say("Cognitive Linguistic Engagement Operator now Active. Please state your message!")
tts_engine.runAndWait()

# Interactive loop
while True:
    try:
        # Capture speech
        with mic as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source)  # Noise reduction
            audio = recognizer.listen(source)

        # Convert speech to text
        prompt = recognizer.recognize_google(audio).strip()
        print(f"You: {prompt}")

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            tts_engine.say("Sayonara")
            tts_engine.runAndWait()
            break

        # System instruction for concise answers
        system_message = "C.L.E.O Active, Please state your message. Keep responses short and precise."
        input_text = f"{system_message}\nUser: {prompt}"

        # Tokenize input and add attention mask
        model_inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100  # Limit input size
        ).to(device)

        if tokenizer.pad_token_id is not None:
            model_inputs["attention_mask"] = (model_inputs.input_ids != tokenizer.pad_token_id).long()

        # Generate response with constraints
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=50,  # Limit response length
            temperature=0.7,  # More focused responses
            top_p=0.8  # Encourages shorter, relevant outputs
        )

        # Decode and clean response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove unwanted prefixes (Assistant:, AI:, Chatbot:, etc.)
        response_cleaned = re.sub(r"^(Assistant:|AI:|Chatbot:)?", "", response, flags=re.IGNORECASE).strip()

        print(f"\nChatbot: {response_cleaned}")

        # Convert response to speech
        tts_engine.say(response_cleaned)
        tts_engine.runAndWait()

    except sr.UnknownValueError:
        print("Could not understand audio, please try again.")
        tts_engine.say("I didn't catch that, please repeat.")
        tts_engine.runAndWait()
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        tts_engine.say("Speech recognition service is unavailable.")
        tts_engine.runAndWait()