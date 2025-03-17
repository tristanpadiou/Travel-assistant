from travel_agent import travel_agent
import gradio as gr
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
load_dotenv()

GOOGLE_API_KEY=os.getenv('google_api_key')
GEMINI_MODEL='gemini-2.0-flash'
llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_MODEL, temperature=0.3)

travel_assistant=travel_agent(llm)

def chat(input, history):
    response=travel_assistant.chat(input)
    return response

#initializing the agent
with gr.Blocks(title="Travel Assistant") as app:
    gr.Markdown("# Travel Assistant")
    gr.Markdown("Upload an image of your schedule, then ask questions about it.")
    
    
    with gr.Row():
        with gr.Sidebar():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Schedule Image", type="pil")
                upload_button = gr.Button("Process Schedule")
            
        with gr.Column(scale=2):
            gr.ChatInterface(chat, type="messages", autofocus=False, )
    
    upload_button.click(
        fn=travel_assistant.image_processing,
        inputs=image_input
    )
    
    # message.submit(
    #     fn=chat,
    #     inputs=[message, chatbot],
    #     outputs=[chatbot]
    # )
    
    gr.Markdown("## Example Questions")
    gr.Markdown("""
    - "Show me my full schedule"
    - "What's on my schedule for Monday?"
    - "Do I have anything at 2:00 PM?"
    - "When is my team meeting?"
    - "What notes are in my schedule?"
    - "When am I free on Tuesday?"
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()