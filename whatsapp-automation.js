const { Client } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');


const specifiedNumber = '910000000000@c.us'; // Replace with the actual number with 91

const client = new Client();

// txt file banane ka function
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

// QR ke liye
client.on('qr', (qr) => {
    console.log("QR Code generated:");
    qrcode.generate(qr, { small: true });
});

// Client ready message
client.on('ready', () => {
    console.log("Client is ready!");
});

// Incoming message cheking dono case ke liye 
client.on('message', message => {
    console.log("Received a message"); // Debugging log
    if (message.from === specifiedNumber) {
        console.log(`New message from ${message.from}: ${message.body}`);
        updateMessageFile(message.from, message.body);
    } else {
        console.log(`Message from another number: ${message.from}`); // Debugging log
    }

// Fetching all messages
// console.log(`New message from ${message.from}: ${message.body}`);
// updateMessageFile(message.from, message.body);

});

// Logout event message ke liye
client.on('disconnected', (reason) => {
    console.log(`Client was logged out. Reason: ${reason}`);
    fs.appendFile('messages.txt', `Client logged out at ${new Date().toLocaleString()}\n`, (err) => {
        if (err) {
            console.error("Error writing logout message to file", err);
        } else {
            console.log("Logout message saved to messages.txt");
        }
    });
});

// Error handling
client.on('error', (error) => {
    console.error("An error occurred:", error);
});


client.initialize();
