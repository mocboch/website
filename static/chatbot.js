
var seq = 0
function loadChatHistory() {

    $.get('/get_chat_history', function(data) {
        data.history.forEach(function(message) {
            if (message.role === 'model') {
                addMessage('Mr. Botner: ' + message.parts[0].text);
            }
            else {
                addMessage('You: ' + message.parts[0].text);
            }
        });
    });
}

function sendMessage() {
    if (seq === 1) {
        var userInput = document.getElementById('userInput');
        var message = userInput.value;
        userInput.value = '';
        addMessage('Note for Mark: ' + message);
        fetch('/handleMemo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({message: message}),
        }).then(seq = 0).then(addMessage('Mr. Botner: Ok, I\'ll let Mark know. What else can I help you with?'))
    }
    else {
        var userInput = document.getElementById('userInput');
        var message = userInput.value;
        if (message.trim() !== '') {
            addMessage('You: ' + message);
            userInput.value = '';
            // Send the message to the server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message}),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.type === 2) {
                        window.location.href = data.redirect_url;
                    }
                    addMessage('Mr. Botner: ' + data.response);
                    if (data.type === 3) {
                        seq = 1;
                    }
                    if (data.type === 4) {
                        openCalendly();
                    }

                });
        }
    }
}

function addMessage(message) {
    var chatMessages = document.getElementById('chatMessages');
    var messageElement = document.createElement('p');
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function openCalendly() {
            const browser = document.getElementById('calendly');
            const content = document.getElementById('calendly-content');

            browser.style.display = 'block';
        }
function closeCalendly() {
    const browser = document.getElementById('calendly');
    browser.style.display = 'none';
}
window.onload = function() {
    closeCalendly();
    Calendly.initInlineWidget({
        url: 'https://calendly.com/markbochner1',
        parentElement: document.getElementById('calendly-embed')
    });
};