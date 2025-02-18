from travel_agent import travel_agent
import gradio as gr
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
load_dotenv()

GOOGLE_API_KEY=os.getenv('google_api_key')
GEMINI_MODEL='gemini-2.0-flash'
llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_MODEL, temperature=0.3)

#initializing the agent
travel_ai=travel_agent(llm)


def chatbot(input, history):
    #no need for history since agent has state memory already
    response=travel_ai.chat(input)
    return response
demo = gr.ChatInterface(chatbot, type="messages", autofocus=False)

if __name__ == "__main__":
    demo.launch()