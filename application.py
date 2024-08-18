from flask import *
import google.generativeai as genai
from flask_pymongo import PyMongo
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from copy import deepcopy as copy
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import threading
from user_agents import parse
from GoogleEmbeddings import Embeddings
from datetime import datetime

application = app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.langchain_api_key = open('langchain_api_key.txt').read()


def checkRefresh(timestamp):
    if (datetime.now() - timestamp).seconds > 3:
        refresh_chat()
def is_mobile(request):
    user_agent = parse(request.headers.get('User-Agent'))
    return user_agent.is_mobile


@app.route('/refresh_chat', methods=['POST'])
def refresh_chat():

    def inputUserData(name: str=None,
                      email: str=None,
                      how_found: str=None,
                      is_recruiter: bool=None,
                      company_name: str=None,
                      job_hiring_for: str=None,
                      job_description: str=None):
        '''
        Extracts data from a chat log. Called when asked to extract data from a chat log.
        Args:
            name: the name of the user, or None if the user has not provided their name.
            email: the user's email address, or None if the user has not provided their email address
            how_found: how the user was directed to the website, or None if the user has not provided this information
            is_recruiter: A boolean variable representing whether this person is a recruiter, or None if the user has not provided this information
            company_name: the name of the company the user works for, or None if the user has not provided this information
            job_hiring_for: the job the user is hiring for, or None if the user has not provided this information
            job_description: a brief description of the job if the user provides the information; otherwise None
        '''
        return {name:name,
                email:email,
                how_found:how_found,
                is_recruiter:is_recruiter,
                company_name:company_name,
                job_hiring_for:job_hiring_for,
                job_description:job_description}

    summary_model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=[
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
        ], tools=[inputUserData])
    chat_history = session.get('chat_hist', [{'role': 'model',
                                              'parts': [{'text': 'Hi there! I\'m Mark\'s AI assistant. How can I help you?'}]}])

    chat_history_string = ''

    chat_history_string = [f'{chat_history_string}  {message["role"]}: {message["parts"][0]["text"]}' for message in chat_history]
    response = summary_model.generate_content(f'Extract data from the following chat log, by calling your function:'
                                            f'{chat_history_string}')
    args = {}
    print(response)
    for key in list(response.parts[0].function_call.args.keys()):  # serializes the arguments from gemini as a list
        args[key] = response.parts[0].function_call.args[key]
    doc = locals()[response.parts[0].function_call.name](**args)


    client = MongoClient(app.mongo_uri)
    db = client['website-database']
    collection = db['chat_logs']
    clean_doc = {key: args[key] for key in args.keys() if key is not None}
    clean_doc['log'] = chat_history_string
    clean_doc['timestamp'] = datetime.now().isoformat()
    #print(clean_doc)
    result = collection.insert_one(clean_doc)
    session['chat_hist'] = [{'role': 'model',
                             'parts': [{'text': 'Hi there! I\'m Mark\'s AI assistant. How can I help you?'}]}]  # chat_hist is stored as a list of dicts in session memory
    return jsonify({'reload': 1})

@app.before_request
def startup():                                                                      #Sets up chat and vectorstore on startup
    app.retriever_ready = False
    app.projects_retriever_ready = False
    app.is_first_refresh = True
    app.current_page = 'home'

    with open('mongo_info.txt') as f:
        (user, password, url) = f.readlines()                                       #Assemble the mongo connection URI
    mongo_uri = f'mongodb+srv://{user.strip()}:{password.strip()}@{url.strip()}/?retryWrites=true&w=majority&appName=website-database'
    app.mongo_uri = mongo_uri
    app.config["MONGO_URI"] = os.environ.get('MONGODB_URI', mongo_uri)
    mongo = PyMongo(app)                                                            #Configure and run PyMongo
    app.google_api_key = open('google_api_key.txt').read()
    genai.configure(api_key=app.google_api_key)

    refresh = refresh_chat()
    def create_retriever(mongo_uri=mongo_uri):
        embeddings = Embeddings(api_key=app.google_api_key)                                         #Instantiate a HuggingFaceEmbeddings model- this is taking too long
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
            'website-database.education-v2',                                               #Create a vector search object
            embeddings,
            index_name="vector_index"
        )
        app.retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 15})    #Store it as a retriever for use later
        app.retriever_ready = True

    threading.Thread(target=create_retriever, daemon=True).start()
    app.before_request_funcs[None].remove(startup)

def before_request():
    if app.config.get('PREFERRED_URL_SCHEME', 'http') == 'https':
        from flask import _request_ctx_stack
        if _request_ctx_stack is not None:
            reqctx = _request_ctx_stack.top
            reqctx.url_adapter.url_scheme = 'https'


@app.route('/')                                             #Home page- robot image and chatbar
def home():
    print('/home called')
    if is_mobile(request):
        return render_template('mobile_resume.html')
    else:
        return render_template('home.html')

@app.route('/projects')                                             #Projects and Github browser page
def projects():
    print('/projects called')
    app.current_page = 'projects'
    if is_mobile(request):
        return render_template('mobile_projects.html')
    else:
        return render_template('projects.html')

@app.route('/Resume', methods=['GET'])              #Dynamic resume page
def Resume():
    print('/resume called')
    app.current_page = 'resume'
    if is_mobile(request):
        return render_template('mobile_resume.html')
    else:
        return render_template('Resume.html')

@app.route('/mobile_contact', methods=['GET'])
def contact():
    return render_template('mobile_contact.html')
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
    app.last_chat = datetime.now()
    threading.Thread(target=checkRefresh(app.last_chat), daemon=True).start()
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
        client = MongoClient(app.mongo_uri)
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

    def goHome(text: str=None):
        '''
        Returns the user to the homepage. Called when the user requests to return to the home page.
        Args:
            text: ignore this argument
        Returns:
            rendered template
        '''
        print('goHome called')
        if text is None or text.strip() == '':
            text = session['chat_hist'][-1]['parts'][0]['text'] = 'No problem. Is there anything else I can help you with today?'
        else:
            text = session['chat_hist'][-1]['parts'][0]['text'] = text
        return jsonify({'response': text, 'redirect_url': url_for('home'), 'type': 2})

    def showProjects(text: str=None):
        '''
        Shows Mark's projects to the used. Called when the user requests to see Mark's projects, or asks about Mark's project experience.
        Args:
            text: ignore this argument
        Returns:
            rendered template showing Mark's projects
        '''
        print('showProjects called')
        if text is None or text.strip() == '':
            text = session['chat_hist'][-1]['parts'][0]['text'] = 'Mark has a few projects available on his github; you can browse the highlights here!'
        else:
            session['chat_hist'][-1]['parts'][0]['text'] = text
        return jsonify({'response': text, 'redirect_url': url_for('projects'), 'type': 2})
    def showResume(job_title: str=None, text: str=None):
        '''
        Shows Mark's Resume page to users. Called whenever a user inquires about Mark's resume.
        Args:
            text: ignore this argument
            job_title: A string containing the job title or description. If none is available pass None.
        Returns:
            rendered template
        '''
        print('showResume called')
        if job_title is None:
            if text is None or text.strip() == '':
                text = session['chat_hist'][-1]['parts'][0]['text'] = 'Sure, here is Mark\'s resume'
            else:
                session['chat_hist'][-1]['parts'][0]['text'] = text
            category = 'none'
        else:
            category = None
            if 'student' in job_title.lower() or 'intern' in job_title.lower() or 'science' in job_title.lower() or 'scientist' in job_title.lower():
                category = 'student'
            elif 'apprentice' in job_title.lower():
                category = 'mentorship'
            elif 'engineer' in job_title.lower() or 'integration' in job_title.lower() or ' AI ' in job_title or ' ML ' in job_title:
                category = 'engineer'
            elif 'manager' in job_title.lower() or 'leader' in job_title.lower():
                category = 'leader'
            elif 'data' in job_title.lower():
                category = 'data'
            else:
                if text is None or text.strip() == '':
                    text = session['chat_hist'][-1]['parts'][0]['text'] = 'Sure, here is Mark\'s resume'
                else:
                    session['chat_hist'][-1]['parts'][0]['text'] = text
                category = 'none'
        return jsonify({'response': text, 'redirect_url': url_for('Resume'), 'type':5, 'category': category, 'job_title':job_title})

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

    def showContact(text: str=None):
        '''
        Provides Mark's contact info to the user. Called when the user asks for Mark's contact information or when the user asks how to get in touch with Mark.
        Args:
            text: ignore this argument
        Returns:
            Mark's contact information
        '''
        if text is None or text.strip() == '':
            text = session['chat_hist'][-1]['parts'][0]['text'] = 'Here\'s Mark\'s contact info; I can also take a message or set up a meeting if you\'d like.'
        else:
            session['chat_hist'][-1]['parts'][0]['text'] = text
        return jsonify({'response': 'Here\'s Mark\'s contact info; I can also take a message or set up a meeting if you\'d like.',
                        'type': 4, 'request': 'contact'})

    def setInterview(text: str=None):
        '''
        Sets an interview with Mark. Called when the user requests to speak with Mark or inquires about his schedule.
        Args:
            text: ignore this argument
        Returns:
            a calendly interface to interact with the customer
        '''
        if text is None or text.strip() == '':
            text = session['chat_hist'][-1]['parts'][0]['text'] = 'I can set up a meeting with Mark- Have a look at his calendar and let me know what works for you.'
        else:
            session['chat_hist'][-1]['parts'][0]['text'] = text
        return jsonify({'response': 'I can set up a meeting with Mark- Have a look at his calendar and let me know what works for you.',
                        'type': 4, 'request': 'calendar'})
    def discussEducation(query: str):
        '''
        Do not ask followup questions before calling this function.
        Returns information about Mark's Education. Called when the user asks for information about Mark's education,
        including the Applied Business Analytics or Salesforce programs at ASU, the MS in Data Science at Eastern University,
        Alfred University, or the Academy for Information Technology.
        Args:
            query: A string, either containing the user's request or a slightly modified version of it if appropriate.
        returns:
            A response to the user's request
        '''
        client = MongoClient(app.mongo_uri)
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
    tools = [showContact, sendMemo, setInterview, goHome, showResume, showProjects]
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
                             'parts': [{'text': 'Hi there! I\'m Mark\'s AI assistant. How can I help you?'}]}])                                         #Get the chat history, which can be fed to the model as a list of dicts
    chat = model.start_chat(history=[{'role': 'user', 'parts': [{'text': 'You are Mr. Botner, Mark\'s AI assistant. You will attend politely to the user\'s requests. Do not be pushy, and make sure to call your tools when appropriate. If an opportunity presents itself, you may ask what role the user is hiring for, at what company, who they are, etc. The user is currently looking at Mark\'s ' + app.current_page + ' page.'}]}] + hist[-10:])                                   #Start a chat with the model and history
    message = request.json['message']                                       #Gets the message (prompt) from the front end
    response = chat.send_message(message)
    print(response)#Calls gemini for a response to the prompt
    bypass_response_functions = [discussEducation]
    if not response.parts[0].function_call and len(response.parts) == 1:                                     #Returns a text based response directly to the front end
        session['chat_hist'] += [{'role': msg.role,                          #Serializes the chat history as a list of dicts and puts it away in the session storage
                                 'parts': [{'text': part.text}
                                           for part in msg.parts]}
                                for msg in chat.history[-2:]]
        return jsonify({'response': response.text, 'type': 1})
    elif len(response.parts) == 1:                                                                   #else handles function calls
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

        return locals()[fn_call](**args)
    else:
        fn_call = response.parts[1].function_call.name  # gets the name of the function called
        args = {}
        for key in list(response.parts[1].function_call.args.keys()):  # serializes the arguments from gemini as a list
            args[key] = response.parts[1].function_call.args[key]
        print(fn_call)
        if fn_call not in bypass_response_functions:
            args['text'] = response.parts[0].text
        h = copy(chat.history[0])  # Makes a deepcopy of the first message, which is user submitted and therefore text
        chat.history[-1] = h  # Uses that as a blank to fill in the appropriate response once it's generated
        chat.history[-1].parts[0].text = ' '
        chat.history[-1].role = 'model'
        session['chat_hist'] += [{'role': msg.role,  # Serialize and store the chat history
                                  'parts': [{'text': part.text}
                                            for part in msg.parts]}
                                 for msg in chat.history[-2:]]

        return locals()[fn_call](**args)
@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():                                                     #route to update chat history after changing pages
    hist = session.get('chat_hist', [{'role': 'model',
                             'parts': [{'text': 'Hi there! I\'m Mark\'s AI assistant. How can I help you?'}]}]  # chat_hist is stored as a list of dicts in session memory
)
    return jsonify({'history': hist})



if __name__ == '__main__':
    app.run(debug=True, port=8000)