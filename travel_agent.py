from maps_agent import maps_agent
from schedule_agent import Schedule_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import load_tools
from langchain.tools import Tool,tool,StructuredTool

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    HumanMessage,
)
from pydantic import BaseModel, Field
import pytz
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict
from typing import Annotated
#get graph visuals
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
import os
import requests
from dotenv import load_dotenv 
load_dotenv()


GOOGLE_API_KEY=os.getenv('google_api_key')
pse=os.getenv('pse')
OPENWEATHERMAP_API_KEY=os.getenv('open_weather_key')
os.environ['OPENWEATHERMAP_API_KEY']=OPENWEATHERMAP_API_KEY

GEMINI_MODEL='gemini-2.0-flash'
llm = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GEMINI_MODEL, temperature=0.3)

class State(TypedDict):
    messages:Annotated[list, add_messages]

maps_ai=maps_agent(llm)
schedule_ai=Schedule_agent(llm)


# extra tools for question answering

# initializing time and date tool

#creating a schema
class time_tool_schema(BaseModel):
  continent: str = Field(description='continent')
  city: str = Field(description='city')

def date_time_tool(continent: str,city: str) -> str:
  """
  tool to get the current date and time in a city.

  """
  city=city.replace(' ','_')
  continent=continent.replace(' ','_')
  query=continent+'/'+city
  timezone = pytz.timezone(query)
  # Get the current time in UTC, and then convert it to the Marrakech timezone
  utc_now = datetime.now(pytz.utc)  # Get current time in UTC
  localized_time = utc_now.astimezone(timezone)  # Convert to Marrakech time
  time=localized_time.strftime('%Y-%m-%d %H:%M:%S')
  return time

current_date_time_tool=StructuredTool.from_function(name='current_date_time_tool', func=date_time_tool, description='To get the current date and time in any city',args_schema=time_tool_schema, return_direct=True)

def google_image_search(query: str) -> str:
  """Search for images using Google Custom Search API
  args: query
  return: image url
  """
  # Define the API endpoint for Google Custom Search
  url = "https://www.googleapis.com/customsearch/v1"

  params = {
      "q": query,
      "cx": pse,
      "key": GOOGLE_API_KEY,
      "searchType": "image",  # Search for images
      "num": 1  # Number of results to fetch
  }

  # Make the request to the Google Custom Search API
  response = requests.get(url, params=params)
  data = response.json()

  # Check if the response contains image results
  if 'items' in data:
      # Extract the first image result
      image_url = data['items'][0]['link']
      return image_url
  else:
      return "Sorry, no images were found for your query."

google_image_tool=Tool(name='google_image_tool', func=google_image_search, description='Use this tool to search for images using Google Custom Search API')

@tool
def schedule_manager(query:str):
    """
    Use this tool for any schedule related queries
    this tool can: 
    list the local files
    load a schedule
    make edits to the schedule
    answer questions about the schedule
    save the schedule
    args:query - pass the schedule related queries directly here
    """
    response=schedule_ai.chatbot(str(query))
    return response

@tool
def maps_tool(query: str):
    """
    Use this tool for any maps or location related queries
    all the context is provided in the tool, simply pass the query
    this tool can:
    get the current location
    find nearby places
    find places in different locations
    show the places that have been found
    args:query - maps or location related queries
    """
    response=maps_ai.chatbot(str(query))
    return response

class travel_agent:
    def __init__(self,llm: any):
        self.agent=self._setup(llm)
        

    def _setup(self,llm):
        api_tools=load_tools(['openweathermap-api','wikipedia'])
        langgraph_tools=[current_date_time_tool,google_image_tool,schedule_manager,maps_tool]+api_tools


        graph_builder = StateGraph(State)
        
        # Modification: tell the LLM which tools it can call
        llm_with_tools = llm.bind_tools(langgraph_tools)
        tool_node = ToolNode(tools=langgraph_tools)
        def chatbot(state: State):
            """ travel assistant that answers user questions about their trip.
            Depending on the request, leverage which tools to use if necessary."""
            return {"messages": [llm_with_tools.invoke(state['messages'])]}

        graph_builder.add_node("chatbot", chatbot)


        graph_builder.add_node("tools", tool_node)
        # Any time a tool is called, we return to the chatbot to decide the next step
        graph_builder.set_entry_point("chatbot")

        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        memory=MemorySaver()
        graph=graph_builder.compile(checkpointer=memory)
        return graph
    
    def display_graph(self):
        return display(
            Image(
                    self.agent.get_graph().draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                    )
                )
            )
    def get_state(self, state_val:str):
        config = {"configurable": {"thread_id": "1"}}
        return self.agent.get_state(config).values[state_val]
    
    def stream(self,input:str):
        config = {"configurable": {"thread_id": "1"}}
        input_message = HumanMessage(content=input)
        for event in self.agent.stream({"messages": [input_message]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()

    def chatbot(self,input:str):
        config = {"configurable": {"thread_id": "1"}}
        response=self.agent.invoke({'messages':HumanMessage(content=str(input))},config)
        return response['messages'][-1].content