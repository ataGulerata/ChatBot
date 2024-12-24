from flask import Flask, request, jsonify, render_template
from chatbot.chatbot import Chatbot

class ChatbotApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.chatbot = Chatbot()
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/get_response', methods=['POST'])
        def get_response():
            user_input = request.json['message']
            response = self.chatbot.get_response(user_input)
            return jsonify({'response': response})

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    app = ChatbotApp()
    app.run()
