const chatContainer = document.getElementById('chat-container');
const messageInput = document.querySelector('textarea');
const sendButton = document.querySelector('.btn-primary');
const cards = document.querySelectorAll('.card');
const cardsContainer = document.querySelector('.cards');
const heading = document.querySelector('.heading');
const button = document.querySelector('.btn');


// Function to create a message element
function createMessageElement(content, alignment, isHTML = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${alignment}`;

    if (isHTML) {
        messageDiv.innerHTML = content;
    } else {
        messageDiv.textContent = content;
    }

    return messageDiv;
}

// Existing code...

// Update this function to automatically scroll to the bottom
function updateScroll() {
    chatContainer.scrollTop = chatContainer.scrollHeight - chatContainer.clientHeight;
}

// Call updateScroll when the page loads to show the latest messages
window.onload = updateScroll;

// ... rest of the existing code


let firstMessageSent = false; // Add this line at the start of your script to track if the first message has been sent

function sendRight() {
    const text = messageInput.value.trim();
    if (text !== '') {
        const messageElement = createMessageElement(text, 'right');
        chatContainer.appendChild(messageElement);
        messageInput.value = '';
        updateScroll();

        if (!firstMessageSent) {
            heading.style.display = 'none';
            cardsContainer.style.display = 'none';
            firstMessageSent = true;
        }

        // Create a "loading" message element with HTML content and append it to the chatContainer
        const loadingContent = `
            <p class="card-text placeholder-glow">
                <span class="placeholder col-7"></span>
                <span class="placeholder col-4"></span>
                <span class="placeholder col-4"></span>
                <span class="placeholder col-6"></span>
            </p>`;
        const loadingMessage = createMessageElement(loadingContent, 'left', true);
        loadingMessage.style.width = '160px'
        chatContainer.appendChild(loadingMessage);
        updateScroll();

        // URL for the API endpoint
        const url = 'https://52aa-54-160-246-180.ngrok-free.app/chat';

        // Data to be sent in the POST request
        const data = {
            user_input: text,
        };

        button.classList.toggle('disabled');
        // Making the POST request
        fetch(url, {
            method: 'POST', // Specifying the HTTP method
            headers: {
                'Content-Type': 'application/json' // Indicating the type of content being sent
            },
            body: JSON.stringify(data) // Converting the data to a JSON string
        })
            .then(response => response.json()) // Parsing the JSON response
            .then(data => {
                chatContainer.removeChild(loadingMessage);
                console.log('Success:', data); // Printing the response data
                if (data.department_courses_recommendations.includes('No specific department courses found based on the query.')) {
                    sendLeft(data.other_queries_responses);
                }
                else {
                    sendLeft(data.department_courses_recommendations + "<br>" + data.other_queries_responses);
                }
            })
            .catch((error) => {
                chatContainer.removeChild(loadingMessage);
                console.error('Error:', error); // Printing any errors that occur
                sendLeft('Something went wrong. Please try again.'); // Sending a user-friendly error message
            })
            .finally(() => {
                button.disabled = false; // Re-enable the button whether there was an error or not
            });
    }
}

function typeWriter(text, element, callback) {
    let index = 0;
    function write() {
        if (index < text.length) {
            if (text.substr(index, 4) === '<br>') {
                element.appendChild(document.createElement('br'));
                index += 4;
            } else {
                element.appendChild(document.createTextNode(text[index]));
                index++;
            }
            setTimeout(write, 20); // adjust the timeout for typing speed
        } else {
            callback();
        }
    }
    write();
}




function sendLeft(text) {
    text = text.toString().replace(/\n/g, '<br>'); // Converting text to string here
    const messageElement = createMessageElement('', 'left', true);
    chatContainer.appendChild(messageElement);

    // Assuming typeWriter sets innerText or textContent, modify it to set innerHTML
    typeWriter(text, messageElement, () => {
        updateScroll();
        button.classList.toggle('disabled');
    });
}



messageInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent the default action (new line) when Enter key is pressed
        sendRight();
    }
});

// Attach event listener to the send button
sendButton.addEventListener('click', sendRight);

cards.forEach(card => {
    card.addEventListener('click', () => {
        const text = card.querySelector('.card-text').textContent.trim();
        messageInput.value = text;  // Set the text content of the card to the message input value
        sendRight(); // Call your existing sendRight function
        cardsContainer.style.display = 'none'; // Hide the entire cards container after sending the message
        heading.style.display = 'none';
    });
});
