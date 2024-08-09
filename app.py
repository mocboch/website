from flask import *
import google.generativeai as genai
from flask_pymongo import PyMongo
from bson import ObjectId
from uuid import uuid4
import os
import pickle
from copy import deepcopy as copy
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sqlalchemy.sql.functions import current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


@app.before_request
def startup():
    session['chat_hist'] = []
    with open('mongo_info.txt') as f:
        (user, password, url) = f.readlines()
    mongo_uri = f'mongodb+srv://{user.strip()}:{password.strip()}@{url.strip()}/?retryWrites=true&w=majority&appName=website-database'
    session['mongo_uri'] = mongo_uri
    app.config["MONGO_URI"] = os.environ.get('MONGODB_URI', mongo_uri)
    mongo = PyMongo(app)
    embeddings = HuggingFaceEmbeddings()
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        mongo_uri,
        'website-database.education',
        embeddings,
        index_name="vec_ind"
    )
    app.retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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
    def discussEducation(query: str):
        '''
        Returns information about the Advanced Business Data Analytics program
        Args:
            query: A string, either containing the user's request or a slightly modified version of it if appropriate.
        returns:
            A response to the user's request
        '''
        google_api_key = open('google_api_key.txt').read()
        client = MongoClient(session['mongo_uri'])
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=google_api_key)
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": app.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )
        r = ' '.join([chunk for chunk in rag_chain.stream(query)])
        return jsonify({'response': r, 'type':1})

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
        ], tools=[showResume, discussEducation])
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
        args = {}
        for key in list(response.parts[0].function_call.args.keys()):
            args[key] = response.parts[0].function_call.args[key]
        h = copy(chat.history[0])
        chat.history[-1] = h
        chat.history[-1].parts[0].text = 'Sure, here is Mark\'s Resume'
        chat.history[-1].role = 'model'
        session['chat_hist'] = [{'role': msg.role,
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history]
        return locals()[fn_call](**args)#Figure out how to pull args

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    hist = session.get('chat_hist', [])
    return jsonify({'history': hist})

if __name__ == '__main__':
    app.run(debug=True)