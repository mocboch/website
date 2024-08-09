from flask import *
import google.generativeai as genai
from uuid import uuid4
import os
import pickle
from copy import deepcopy as copy

from sqlalchemy.sql.functions import current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


@app.before_request
def startup():
    session['chat_hist'] = []

    app.before_request_funcs[None].remove(startup)


@app.route('/')
def home():

    return render_template('home.html')
@app.route('/Resume', methods=['GET'])
def Resume():
    return render_template('Resume.html')

'''
@app.route('/refresh', methods=['POST'])
def handle_refresh():
    session['chat_hist'] = []
    return home()
'''

@app.route('/chat', methods=['POST'])
def chat():
    with open('google_api_key.txt') as f:
        genai.configure(api_key=f.read())
    def showResume(jobdesc: str):
        '''
        Shows Mark's Resume page to any user who requests it.
        Args:
            jobdesc: A string containing the job title or description. If none is available pass None.
        Returns:
            rendered template
        '''
        print('redir' + 'jobdesc=' + jobdesc)
        return jsonify({'response': 'Sure, here is Mark\'s resume', 'redirect_url': url_for('Resume'), 'type':2})
    model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ], tools=[showResume])
    hist = session.get('chat_hist')
    chat = model.start_chat(history=hist)
    message = request.json['message']
    response = chat.send_message(message)
    if len(response.parts[0].text) > 0:
        session['chat_hist'] = [{'role': msg.role,
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history]
        return jsonify({'response': response.text, 'type': 1})
    else:
        fn_call = response.parts[0].function_call.name
        h = copy(chat.history[0])
        chat.history[-1] = h
        chat.history[-1].parts[0].text = 'Sure, here is Mark\'s Resume'
        chat.history[-1].role = 'model'
        session['chat_hist'] = [{'role': msg.role,
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history]
        return locals()[fn_call](jobdesc='asfd')#Figure out how to pull args

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    hist = session.get('chat_hist', [])
    return jsonify({'history': hist})

if __name__ == '__main__':
    app.run(debug=True)