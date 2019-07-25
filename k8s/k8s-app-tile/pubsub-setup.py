from google.cloud import pubsub_v1 as pubsub
import os

project_id = os.environ['project_id']
topic_name = os.environ['topic_name']
subscription_name = os.environ['subscription_name']
ack_deadline_seconds = int(os.environ['ack_deadline_seconds'])

publisher = pubsub.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)
subscriber = pubsub.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_name)

def delete_subscription(subscription_path):
    """Deletes an existing Pub/Sub subscription."""
    subscriber.delete_subscription(subscription_path)
    print('Subscription deleted: {}'.format(subscription_path))

# Cleaning up if topic_path already exists and clean_topic flag is on
if (topic_path in [top.name for top in publisher.list_topics(publisher.project_path(project_id))]):
    # Deleting existing Pub/Sub subscriptions
    [delete_subscription(sub_path) for sub_path in publisher.list_topic_subscriptions(topic_path)]

    # Deleting an existing Pub/Sub topic
    publisher.delete_topic(topic_path)
    print('Topic deleted: {}'.format(topic_path))

# Create a brand new topic
publisher.create_topic(topic_path)
print('Topic created: {}'.format(topic_path))

# Creating subscription if already doesn't exist
if not subscription_path in [sub for sub in publisher.list_topic_subscriptions(topic_path)]:
    subscriber.create_subscription(subscription_path, topic_path, ack_deadline_seconds=ack_deadline_seconds)
    print('Subscription created: {}'.format(subscription_path))
