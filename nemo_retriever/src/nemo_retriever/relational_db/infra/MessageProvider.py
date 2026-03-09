from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic

from .SecretsManager import SecretsManager
import json
import os
import logging

logger = logging.getLogger("MessageProvider")


def create_topics(config):
    admin = AdminClient(config)

    new_topics = [
        NewTopic(topic, num_partitions=6, replication_factor=1)
        for topic in ["activity", "recommendation", "insight", "file"]
    ]
    admin_topics = admin.create_topics(new_topics)

    # Wait for each operation to finish.
    for topic, admin_topic in admin_topics.items():
        try:
            admin_topic.result()
            logger.info("Topic {} created".format(topic))
        except Exception:
            pass


class MessageProducer:
    def produce_message(self, topic, msg):
        try:
            self.producer.produce(topic, json.dumps(msg))
            self.producer.flush()
        except Exception as err:
            logger.error("Message delivery failed: {}".format(err))

    def __init__(self, config):
        self.producer = Producer(config)
        self.config = config


class MessageConsumer:
    def __init__(
        self, config, group_id="illumex-consumer", auto_offset_reset="earliest"
    ):
        consumer_config = dict(config)
        consumer_config.update(
            {
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": False,
            }
        )
        self.consumer = Consumer(consumer_config)

    def consume_messages(self, topics, on_message, timeout=1.0, should_stop=None):
        if isinstance(topics, str):
            topics = [topics]
        self.consumer.subscribe(topics)
        while True:
            if callable(should_stop) and should_stop():
                break
            msg = self.consumer.poll(timeout=timeout)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: {}".format(msg.error()))
                continue
            try:
                payload = json.loads(msg.value())
            except Exception:
                payload = msg.value()
            try:
                on_message(msg.topic(), payload, msg)
                self.consumer.commit(asynchronous=True)
            except Exception as err:
                logger.error("Message handling failed: {}".format(err))


if os.environ["LMX_ENV"] != "development":
    configuration = SecretsManager().get_secret_dict(os.environ["LMX_KAFKA_SECRET"])
    config = {
        "bootstrap.servers": configuration["broker"],
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": configuration["username"],
        "sasl.password": configuration["password"],
    }
else:
    # If we want to connect to managed kafka
    if "LMX_KAFKA_PASSWORD" in os.environ:
        config = {
            "bootstrap.servers": os.environ["LMX_KAFKA_BROKER"],
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": os.environ["LMX_KAFKA_USERNAME"],
            "sasl.password": os.environ["LMX_KAFKA_PASSWORD"],
        }
    else:
        broker = os.environ["LMX_KAFKA_BROKER"]
        config = {"bootstrap.servers": broker}
        create_topics(config)


try:
    message_producer = MessageProducer(config)
    message_consumer = MessageConsumer(config)
except Exception as err:
    logger.error("Failed to initialize MessageProvider: {}".format(err))
    message_producer = None
    message_consumer = None


def get_message_producer():
    return message_producer


def get_message_consumer():
    return message_consumer
