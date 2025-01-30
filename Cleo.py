from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3  # Text-to-Speech Library
import speech_recognition as sr  # Speech-to-Text Library
import re  # For response cleaning
import threading  # For running speech recognition in parallel

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)
tts_engine.setProperty("volume", 1.0)

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

# Store conversation history with a limited context window
conversation_history = []  # Stores last few exchanges
history_limit = 3  # Keeps only the last 3 user-bot exchanges
stop_tts = False  # Flag to stop TTS when 'stop' is detected

# Function to monitor speech while TTS is playing
def listen_for_stop():
    global stop_tts
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=1)
                detected_text = recognizer.recognize_google(audio).strip().lower()
                if detected_text == "stop":
                    stop_tts = True
                    break
            except sr.UnknownValueError:
                continue  # Ignore unrecognized speech
            except sr.RequestError:
                print("Speech recognition service unavailable.")
                break
            except sr.WaitTimeoutError:
                continue  # Prevent crash

# Interactive loop
while True:
    try:
        # Capture speech
        with mic as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        # Convert speech to text
        prompt = recognizer.recognize_google(audio).strip()
        print(f"You: {prompt}")

        if prompt.lower() in ["exit", "quit"]:
            print("Exiting... Goodbye!")
            tts_engine.say("Sayonara")
            tts_engine.runAndWait()
            break

        # Update conversation history, keeping only the last few exchanges
        conversation_history.append(f"User: {prompt}")
        conversation_history = conversation_history[-(history_limit * 2):]  # Maintain context

        # Construct input for model
        system_message = "C.L.E.O Active, Please state your message. Keep responses short and precise."
        input_text = f"{system_message}\n" + "\n".join(conversation_history)

        # Tokenize input
        model_inputs = tokenizer(
            [input_text], return_tensors="pt", padding=True, truncation=True, max_length=300
        ).to(device)

        if tokenizer.pad_token_id is not None:
            model_inputs["attention_mask"] = (model_inputs.input_ids != tokenizer.pad_token_id).long()

        # Generate response
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.8
        )

        # Decode and clean response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response_cleaned = re.sub(r"(C\.L\.E\.O Active:|Assistant:|Chatbot:)", "", response).strip()
        response_cleaned = response_cleaned.replace(system_message, "").replace(f"User: {prompt}", "").strip()

        print(f"\nChatbot: {response_cleaned}")

        # Start a background thread to listen for "stop"
        stop_tts = False
        stop_thread = threading.Thread(target=listen_for_stop, daemon=True)
        stop_thread.start()

        # Convert response to speech
        tts_engine.say(response_cleaned)
        tts_engine.runAndWait()

        # Stop speaking if "stop" was detected
        if stop_tts:
            print("\nTTS Stopped by User.")
            tts_engine.stop()

        # Update conversation history with chatbot response
        conversation_history.append(f"C.L.E.O: {response_cleaned}")

    except sr.UnknownValueError:
        print("Could not understand audio, please try again.")
        tts_engine.say("I didn't catch that, please repeat.")
        tts_engine.runAndWait()
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        tts_engine.say("Speech recognition service is unavailable.")
        tts_engine.runAndWait()
