from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are an AI assistant. Maintain context while answering questions.

Conversation History:
{context}

User: {question}

AI:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    conversation_history = []
    print("Welcome to C.L.E.O! Say 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("C.L.E.O: Goodbye!")
            break

        # Combine history with the latest question
        context_str = "\n".join(conversation_history[-5:])  # Keep only last 5 interactions
        
        response = chain.invoke({"context": context_str, "question": user_input})

        # Extract response text correctly
        response_text = response if isinstance(response, str) else getattr(response, "content", str(response))

        print("\nC.L.E.O:", response_text)

        # Store formatted conversation
        conversation_history.append(f"User: {user_input}\nAI: {response_text}")

if __name__ == "__main__":
    handle_conversation()
