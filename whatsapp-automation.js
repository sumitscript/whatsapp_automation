// whatsapp-automation.js
const { Client } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const axios = require('axios'); // Import Axios

// List of specified numbers
const specifiedNumbers = ['919893525748@c.us', '918401997567@c.us', '917568119230@c.us'];

const client = new Client();

// Function to update the message log file
function updateMessageFile(from, messageBody) {
    const timestamp = new Date().toLocaleString();
    const logMessage = `From: ${from} | Timestamp: ${timestamp} | Message: ${messageBody}\n`;

    fs.appendFile('messages.txt', logMessage, (err) => {
        if (err) {
            console.error("Error writing to file", err);
        } else {
            console.log("Message saved to messages.txt");
        }
    });
}

// Function to generate response using the new Python API
async function generateResponse(inputText) {
    try {
        const response = await axios.post('http://localhost:8000/chat', {
            message: inputText
        });
        return response.data.response;
    } catch (error) {
        console.error("Error communicating with Python API:", error.message);
        if (error.code === 'ECONNREFUSED') {
            return "I apologize, but I'm unable to connect to my AI brain. Please ensure the backend service is running.";
        }
        throw new Error("An error occurred while generating the response.");
    }
}

// Event when QR code is generated
client.on('qr', (qr) => {
    console.log("QR Code generated:");
    qrcode.generate(qr, { small: true });
});

// Event when client is ready
client.on('ready', () => {
    console.log("Client is ready!");
    console.log("Make sure your Python FastAPI service is running: uvicorn api:app --reload");
});

// Event for incoming messages
client.on('message', async message => {
    console.log("Received a message");

    if (specifiedNumbers.includes(message.from)) {
        console.log(`New message from specified number ${message.from}: ${message.body}`);
        updateMessageFile(message.from, message.body);

        try {
            const chat = await message.getChat();
            await chat.sendStateTyping();

            const response = await generateResponse(message.body);

            await chat.clearState();

            if (response && response.trim()) {
                console.log(`Sending response: ${response}`);
                await message.reply(response);
            } else {
                await message.reply("I apologize, but I couldn't generate a meaningful response. Please try again.");
            }
        } catch (error) {
            console.error("Error generating response:", error);
            await message.reply("I apologize, but I encountered an error while processing your message. Please try again in a moment.");
        }
    } else {
        console.log(`Message from another number: ${message.from}`);
        updateMessageFile(message.from, message.body);
    }
});

client.on('disconnected', (reason) => {
    console.log(`Client was logged out. Reason: ${reason}`);
    fs.appendFile('messages.txt', `Client logged out at ${new Date().toLocaleString()}\n`, (err) => {
        if (err) console.error("Error writing logout message to file", err);
    });
});

client.on('error', (error) => {
    console.error("An error occurred:", error);
});

client.initialize();