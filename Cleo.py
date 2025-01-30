from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3  # Text-to-Speech Library
import speech_recognition as sr  # Speech-to-Text Library
import re  # For response cleaning
import threading  # For running speech recognition in parallel

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

# Store conversation history to improve context
conversation_history = ""
stop_tts = False  # Flag to stop TTS when 'stop' is detected

# Function to monitor speech while TTS is playing
def listen_for_stop():
    global stop_tts
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Noise reduction
        while True:
            try:
                audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                detected_text = recognizer.recognize_google(audio).strip().lower()
                if detected_text == "stop":
                    stop_tts = True
                    break
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                break

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

        # Add user input to conversation history for better context
        conversation_history += f"\nUser: {prompt}"

        # System instruction for concise answers
        system_message = "C.L.E.O Active, Please state your message. Keep responses short and precise."
        input_text = f"{system_message}{conversation_history}"

        # Tokenize input and add attention mask
        model_inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300  # Increased token limit
        ).to(device)

        if tokenizer.pad_token_id is not None:
            model_inputs["attention_mask"] = (model_inputs.input_ids != tokenizer.pad_token_id).long()

        # Generate response with constraints
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=100,  # Increased response length
            temperature=0.7,  # More focused responses
            top_p=0.8  # Encourages shorter, relevant outputs
        )

        # Decode and clean response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove system message and repeated user input
        response_cleaned = response.replace(system_message, "").replace(f"User: {prompt}", "").strip()

        # Further clean assistant prefixes (e.g., "Assistant:", "Chatbot:", "C.L.E.O Active:")
        response_cleaned = re.sub(r"(C\.L\.E\.O Active:|Assistant:|Chatbot:)\s*", "", response_cleaned, flags=re.IGNORECASE).strip()

        print(f"\nChatbot: {response_cleaned}")

        # Start a background thread to listen for "stop"
        stop_tts = False
        stop_thread = threading.Thread(target=listen_for_stop)
        stop_thread.start()

        # Convert response to speech with interruptible playback
        tts_engine.say(response_cleaned)
        tts_engine.runAndWait()

        # Stop speaking if "stop" was detected
        if stop_tts:
            print("\nTTS Stopped by User.")
            tts_engine.stop()

        # Add chatbot's response to conversation history
        conversation_history += f"\nC.L.E.O: {response_cleaned}"

    except sr.UnknownValueError:
        print("Could not understand audio, please try again.")
        tts_engine.say("I didn't catch that, please repeat.")
        tts_engine.runAndWait()
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        tts_engine.say("Speech recognition service is unavailable.")
        tts_engine.runAndWait()
