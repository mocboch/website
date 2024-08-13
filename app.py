from flask import *
import google.generativeai as genai
from flask_pymongo import PyMongo
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from copy import deepcopy as copy
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import threading

application = app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.langchain_api_key = open('langchain_api_key.txt').read()

@app.route('/refresh_chat', methods=['POST'])
def refresh_chat():
    session['chat_hist'] = [{'role': 'model',
                             'parts': [{'text': 'Hi there! \nI\'m Mr. Botner, Mark\'s AI assistant. I can show you his resume, tell you about his skills, or get you in touch with him. I can even help you set up an interview! \n\nWhat can I do for you today?'}]}]  # chat_hist is stored as a list of dicts in session memory
    return jsonify({'reload': 1})

@app.before_request
def startup():                                                                      #Sets up chat and vectorstore on startup
    app.retriever_ready = False
    app.projects_retriever_ready = False
    with open('mongo_info.txt') as f:
        (user, password, url) = f.readlines()                                       #Assemble the mongo connection URI
    mongo_uri = f'mongodb+srv://{user.strip()}:{password.strip()}@{url.strip()}/?retryWrites=true&w=majority&appName=website-database'
    session['mongo_uri'] = mongo_uri                                                #Store it in session memory
    app.config["MONGO_URI"] = os.environ.get('MONGODB_URI', mongo_uri)
    mongo = PyMongo(app)                                                            #Configure and run PyMongo
    app.google_api_key = open('google_api_key.txt').read()
    refresh = refresh_chat()
    def create_retriever(mongo_uri=mongo_uri):
        embeddings = HuggingFaceEmbeddings()                                            #Instantiate a HuggingFaceEmbeddings model- this is taking too long
        '''
        projects_search = MongoDBAtlasVectorSearch.from_connection_string(
            mongo_uri,
            'website-database.projects',
            embeddings,
            index_name='vec_ind'
        )
        app.projects_retriever = projects_search.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        app.projects_retriever_ready = True
        '''
        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            mongo_uri,
            'website-database.education',                                               #Create a vector search object
            embeddings,
            index_name="vec_ind"
        )
        app.retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 15})    #Store it as a retriever for use later
        app.retriever_ready = True

    threading.Thread(target=create_retriever, daemon=True).start()
    app.before_request_funcs[None].remove(startup)


@app.route('/')                                             #Home page- robot image and chatbar
def home():
    print('/home called')
    return render_template('home.html')
@app.route('/projects')                                             #Projects and Github browser page
def projects():
    print('/projects called')
    return render_template('projects.html')
@app.route('/Resume', methods=['GET'])              #Dynamic resume page
def Resume():
    print('/resume called')
    return render_template('Resume.html')

'''
@app.route('/refresh', methods=['POST'])
def handle_refresh():
    session['chat_hist'] = []
    return home()
'''
@app.route('/handleMemo', methods=['POST'])
def handleMemo():
    message = request.json['message']
    session['chat_hist'].append([
        {'role': 'user',
         'parts': {'text': message}},
        {'role': 'model',
         'parts': {'text': 'Ok, I\'ll let Mark know. What else can I help you with?'}}
    ])
    msg = Mail(
        from_email='deskofmarkbotner@gmail.com',
        to_emails='markbochner1@gmail.com',
        subject='Message from Mr. Botner!',
        plain_text_content=message
    )
    sg = SendGridAPIClient(open('Twilio.txt').readlines()[0].strip())
    response = sg.send(msg)



@app.route('/chat', methods=['POST'])
def chat():                                         #Chat response logic
    genai.configure(api_key=app.google_api_key)           #showResume returns a stock messsage and reroutes to the resume page
                                                 #It also updates the session[chat_hist] to give the model the appropriate context
    '''
    def discussProject(query: str):
        ''''''
        Provides information about Mark's project experience to the User. Called when the user requests information about Mark's projects or experience.
        Args:
            query: A string containing the user's query. Copied or lightly modified if necessary
        Returns:
            The requested information about Mark's project experience
        ''''''
        client = MongoClient(session['mongo_uri'])
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=app.google_api_key)
        prompt = hub.pull("project-prompt", api_key=app.langchain_api_key)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": app.projects_retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )
        r = ' '.join([chunk for chunk in rag_chain.stream(query)])
        session['chat_hist'][-1]['parts'][0]['text'] = r
        print(query)
        return jsonify({'response': r, 'type': 1})
    '''

    def goHome():
        '''
        Returns the user to the homepage. Called when the user requests to return to the home page.
        Returns:
            rendered template
        '''
        print('goHome called')
        session['chat_hist'][-1]['parts'][0]['text'] = 'No problem. Is there anything else I can help you with today?'
        return jsonify({'response': 'No problem. Is there anything else I can help you with today?', 'redirect_url': url_for('home'), 'type': 2})

    def showProjects():
        '''
        Shows Mark's projects to the used. Called when the user requests to see Mark's projects, or asks about Mark's project experience.
        Returns:
            rendered template showing Mark's projects
        '''
        print('showProjects called')
        session['chat_hist'][-1]['parts'][0]['text'] = 'Mark has a few projects available on his github; you can browse the highlights here!'
        return jsonify({'response': 'Mark has a few projects available on his github; you can browse the highlights here!', 'redirect_url': url_for('projects'), 'type': 2})
    def showResume(jobdesc: str):
        '''
        Shows Mark's Resume page to any user who requests it.
        Args:
            jobdesc: A string containing the job title or description. If none is available pass None.
        Returns:
            rendered template
        '''
        print('showResume called')
        session['chat_hist'][-1]['parts'][0]['text'] = 'Sure, here is Mark\'s resume'
        return jsonify({'response': 'Sure, here is Mark\'s resume', 'redirect_url': url_for('Resume'), 'type':2})

    #discussEducation is currently called very narrowly for the ABDA for testing purposes.
    #It searches documents related to my education and returns them as context
    #Also updates chat history for the model
    def sendMemo():
        '''
        Leaves a memo for Mark. Called when the user requests contact info or to get in touch with Mark.
        Returns:
            Sends Mark a note
        '''
        session['chat_hist'][-1]['parts'][0]['text'] = 'Sure, I can definitely take a note for Mark! Go ahead and leave your message below, and I\'ll pass it along.'
        return jsonify({'response': 'Sure, I can definitely take a note for Mark! '
                                    'Go ahead and leave your message below, and I\'ll pass it along.', 'type':3})

    def showContact():
        '''
        Provides Mark's contact info to the user. Called when the user asks for Mark's contact information or when the user asks how to get in touch with Mark.
        Returns:
            Mark's contact information
        '''
        session['chat_hist'][-1]['parts'][0]['text'] = 'Here\'s Mark\'s contact info; I can also take a message or set up a meeting if you\'d like.'
        return jsonify({'response': 'Here\'s Mark\'s contact info; I can also take a message or set up a meeting if you\'d like.',
                        'type': 4, 'request': 'contact'})

    def setInterview():
        '''
        Sets an interview with Mark. Called when the user requests to speak with Mark or inquires about his schedule.
        Returns:
            a calendly interface to interact with the customer
        '''
        session['chat_hist'][-1]['parts'][0]['text'] = 'I can set up a meeting with Mark- Have a look at his calendar and let me know what works for you.'
        return jsonify({'response': 'I can set up a meeting with Mark- Have a look at his calendar and let me know what works for you.',
                        'type': 4, 'request': 'calendar'})
    def discussEducation(query: str):
        '''
        Returns information about Mark's Education. Called when the user asks for information about Mark's education,
        including the Applied Business Analytics or Salesforce programs at ASU, the MS in Data Science at Eastern University,
        Alfred University, or the Academy for Information Technology.
        Args:
            query: A string, either containing the user's request or a slightly modified version of it if appropriate.
        returns:
            A response to the user's request
        '''
        client = MongoClient(session['mongo_uri'])
        model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=app.google_api_key)
        prompt = hub.pull('rlm/rag-prompt')

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": app.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )
        r = ' '.join([chunk for chunk in rag_chain.stream(query)])
        session['chat_hist'][-1]['parts'][0]['text'] = r
        print(query)
        return jsonify({'response': r, 'type':1})
    tools = [showContact, showProjects, showResume, sendMemo, setInterview, goHome]
    if app.retriever_ready == True: tools.append(discussEducation)
    #if app.projects_retriever_ready == True: tools.append(discussProject)
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
        ], tools=tools)                                                     #Instantiate a model with tools
    hist = session.get('chat_hist', [{'role': 'model',
                             'parts': [{'text': 'Hi there! \nI\'m Mr. Botner, Mark\'s AI assistant. I can show you his resume, tell you about his skills, or get you in touch with him. I can even help you set up an interview! \n\nWhat can I do for you today?'}]}])                                         #Get the chat history, which can be fed to the model as a list of dicts
    chat = model.start_chat(history=hist[-10:])                                   #Start a chat with the model and history
    message = request.json['message']                                       #Gets the message (prompt) from the front end
    response = chat.send_message(message)                                   #Calls gemini for a response to the prompt
    if len(response.parts[0].text) > 0:                                     #Returns a text based response directly to the front end
        session['chat_hist'] += [{'role': msg.role,                          #Serializes the chat history as a list of dicts and puts it away in the session storage
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history[-2:]]
        return jsonify({'response': response.text, 'type': 1})
    else:                                                                   #else handles function calls
        fn_call = response.parts[0].function_call.name                      #gets the name of the function called
        args = {}
        for key in list(response.parts[0].function_call.args.keys()):       #serializes the arguments from gemini as a list
            args[key] = response.parts[0].function_call.args[key]
        h = copy(chat.history[0])                                           #Makes a deepcopy of the first message, which is user submitted and therefore text
        chat.history[-1] = h                                                #Uses that as a blank to fill in the appropriate response once it's generated
        chat.history[-1].parts[0].text = ' '
        chat.history[-1].role = 'model'
        session['chat_hist'] += [{'role': msg.role,                          #Serialize and store the chat history
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history[-2:]]

        return locals()[fn_call](**args)                                    #return the function call by calling it by name from locals

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():                                                     #route to update chat history after changing pages
    hist = session.get('chat_hist', [{'role': 'model',
                             'parts': [{'text': 'Hi there! \nI\'m Mr. Botner, Mark\'s AI assistant. I can show you his resume, tell you about his skills, or get you in touch with him. I can even help you set up an interview! \n\nWhat can I do for you today?'}]}]  # chat_hist is stored as a list of dicts in session memory
)
    return jsonify({'history': hist})



if __name__ == '__main__':
    app.run(debug=True)