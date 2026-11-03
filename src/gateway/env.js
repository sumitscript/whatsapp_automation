// Must be the first import of the gateway entrypoint: names the service for logs/clients.
process.env.SERVICE_NAME ||= 'gateway';
