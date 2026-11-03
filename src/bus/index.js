// Message bus between the WhatsApp gateway and the AI worker(s).
//   BUS_DRIVER=kafka  → Apache Kafka (KRaft) via kafkajs; key = chatId keeps
//                       per-chat ordering on one partition.
//   BUS_DRIVER=redis  → Redis Streams consumer groups (lighter, no Kafka needed).
// Both expose: publish(topic, key, value), subscribe(topic, group, handler), close().
import { config } from '../core/config.js';

export const TOPICS = Object.freeze({
  inbound: 'wa.inbound', // gateway → worker: reply jobs
  outbound: 'wa.outbound', // worker → gateway: reply bubbles / done / error
});

export async function createBus({ logger, redisFactory }) {
  const driver = config.bus.driver;
  if (driver === 'kafka') return createKafkaBus({ logger });
  if (driver === 'redis') return createRedisBus({ logger, redisFactory });
  throw new Error(`Unknown BUS_DRIVER "${driver}" (use kafka or redis)`);
}

async function createKafkaBus({ logger }) {
  const { Kafka, logLevel, Partitioners } = (await import('kafkajs')).default;
  const log = logger.child({ component: 'kafka' });
  const kafka = new Kafka({
    clientId: `${config.bus.kafkaClientId}-${config.service}`,
    brokers: config.bus.kafkaBrokers,
    logLevel: logLevel.WARN,
    retry: { initialRetryTime: 300, retries: 12 },
    logCreator:
      () =>
      ({ level, log: entry }) => {
        const { message, ...rest } = entry;
        (level <= logLevel.ERROR ? log.error : log.warn)(message, { kafka: rest });
      },
  });

  const admin = kafka.admin();
  await admin.connect();
  try {
    await admin.createTopics({
      waitForLeaders: true,
      topics: Object.values(TOPICS).map((topic) => ({ topic, numPartitions: config.bus.partitions, replicationFactor: 1 })),
    });
  } catch (err) {
    // Gateway and worker boot together and may race to create the same topics.
    if (!/already exists/i.test(String(err.message))) log.warn('Topic creation failed; relying on auto-create', { err });
  } finally {
    await admin.disconnect();
  }

  const producer = kafka.producer({ createPartitioner: Partitioners.DefaultPartitioner, allowAutoTopicCreation: true });
  await producer.connect();
  const consumers = [];
  log.info('Kafka bus connected', { brokers: config.bus.kafkaBrokers });

  return {
    driver: 'kafka',
    async publish(topic, key, value) {
      await producer.send({ topic, acks: 1, messages: [{ key: String(key), value: JSON.stringify(value) }] });
    },
    async subscribe(topic, groupId, handler) {
      const consumer = kafka.consumer({ groupId, sessionTimeout: 30_000, heartbeatInterval: 3000 });
      await consumer.connect();
      await consumer.subscribe({ topic, fromBeginning: false });
      await consumer.run({
        partitionsConsumedConcurrently: config.bus.partitions,
        eachMessage: async ({ message }) => {
          try {
            await handler(JSON.parse(message.value.toString()));
          } catch (err) {
            log.error('Bus handler failed', { topic, err });
          }
        },
      });
      consumers.push(consumer);
    },
    async close() {
      await Promise.allSettled([producer.disconnect(), ...consumers.map((c) => c.disconnect())]);
    },
  };
}

async function createRedisBus({ logger, redisFactory }) {
  const log = logger.child({ component: 'redis-bus' });
  const pub = redisFactory('bus-pub');
  const readers = [];
  let closed = false;

  return {
    driver: 'redis',
    async publish(topic, key, value) {
      await pub.xadd(topic, 'MAXLEN', '~', '10000', '*', 'key', String(key), 'value', JSON.stringify(value));
    },
    async subscribe(topic, group, handler) {
      const reader = redisFactory(`bus-${group}`);
      readers.push(reader);
      try {
        await reader.xgroup('CREATE', topic, group, '$', 'MKSTREAM');
      } catch (err) {
        if (!String(err.message).includes('BUSYGROUP')) throw err;
      }
      const consumer = `${config.service}-${process.pid}`;
      (async () => {
        while (!closed) {
          try {
            const res = await reader.xreadgroup('GROUP', group, consumer, 'COUNT', 20, 'BLOCK', 5000, 'STREAMS', topic, '>');
            if (!res) continue;
            for (const [, entries] of res) {
              for (const [id, fields] of entries) {
                const valueIdx = fields.indexOf('value');
                try {
                  await handler(JSON.parse(fields[valueIdx + 1]));
                } catch (err) {
                  log.error('Bus handler failed', { topic, err });
                }
                await reader.xack(topic, group, id);
              }
            }
          } catch (err) {
            if (closed) break;
            log.warn('Redis stream read failed; retrying', { topic, err });
            await new Promise((r) => setTimeout(r, 1000));
          }
        }
      })();
      log.info('Subscribed to Redis stream', { topic, group });
    },
    async close() {
      closed = true;
      await Promise.allSettled([pub.quit(), ...readers.map((r) => r.disconnect())]);
    },
  };
}
