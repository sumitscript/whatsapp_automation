// Must be the first import of the worker entrypoint: names the service for logs/clients.
process.env.SERVICE_NAME ||= 'worker';
