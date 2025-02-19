# Travel Agent with Schedule and Google Maps Integration

This repository provides a travel agent powered by AI that integrates with a personal schedule and Google Maps. The agent can interact with a schedule, answer location-based queries, and perform Google image searches. It uses the LangChain framework and the Gemini-2.0-Flask model from Google to interpret and respond to various travel-related queries. The agent also allows for real-time time zone conversions for cities around the world.

## Features

- **Schedule Management**: The agent can interact with and manage schedules, including answering questions, loading, editing, and saving schedules.
- **Google Maps Integration**: The agent can help you find places on the map, such as nearby locations, and answer location-based queries.
- **Time Zone Conversion**: Get the current time for cities around the world with the time zone conversion tool.
- **Image Search**: Use Google Custom Search API to find images based on user input.
- **Interactive Interface**: Powered by Gradio, this agent offers a web-based chat interface that simplifies interaction with the AI assistant.

## Installation

### Prerequisites

To run this project, you need to:

- **Google API Key**: For interacting with Google Maps and Google Custom Search API.
- **OpenWeatherMap API Key**: For weather-related queries.
- **Programmable Search Engine ID (PSE)**: For image search functionality.

### Setting up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/travel-agent.git
   cd travel-agent

