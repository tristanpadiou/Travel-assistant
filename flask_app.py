from flask import Flask,render_template, request
from travel_agent import travel_agent

agent_graph=travel_agent()

app = Flask(__name__)
@app.route("/")
def home():    
    return render_template("index.html")
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response = agent_graph.chatbot(userText)  
    #return str(bot.get_response(userText)) 
    return response

if __name__ == "__main__":
   app.run(debug=True)
