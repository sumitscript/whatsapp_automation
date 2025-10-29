# WhatsApp Automation

This is version 1 of the WhatsApp automation project using `whatsapp-web.js`. This project allows you to log in to WhatsApp Web by scanning a QR code and fetch new messages from a specified contact, saving them in a text file.

## Requirements

- Node.js
- npm (Node Package Manager)

## Installation

To install the required packages, run:

```bash
npm install whatsapp-web.js qrcode-terminal
```

### Setup and Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/sumitscript/whatsapp_automation.git
    ```

2. Navigate into the cloned directory:

    ```bash
    cd whatsapp_automation
    ```

3. Run the script:

    ```bash
    node whatsapp-automation.js
    ```

4. Follow the on-screen instructions to scan the QR code for authentication. The application will then start fetching messages from the specified contact and save them to a text file.

## File Creation and Updates

- The application checks for the existence of a `.txt` file for the specified contact.
- If the file does not exist, it creates one and logs the messages.
- If the file exists, it updates the file with new messages along with the timestamp.

## Logout Handling

When you log out from WhatsApp, the application will log the event in the text file.

## License

This project is licensed under the MIT License.

---