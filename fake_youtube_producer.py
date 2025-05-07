import json
import random
import time
import uuid
from kafka import KafkaProducer
from datetime import datetime

# --- Configuration ---
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

regions = ['US', 'CA', 'GB']

# ----------------------------------------------------------------
# 1) Category titles pulled straight from your JSON snippet:
# ----------------------------------------------------------------
categories = [
    "Film & Animation", "Autos & Vehicles", "Music", "Pets & Animals", "Sports",
    "Short Movies", "Travel & Events", "Gaming", "Videoblogging", "People & Blogs",
    "Comedy", "Entertainment", "News & Politics", "Howto & Style", "Education",
    "Science & Technology", "Movies", "Anime/Animation", "Action/Adventure",
    "Classics", "Documentary", "Drama", "Family", "Foreign", "Horror",
    "Sci-Fi/Fantasy", "Thriller", "Shorts", "Shows", "Trailers"
]

# ----------------------------------------------------------------
# 2) Top tags (pipe-separated in payload)
#    pulled from your “Top 40” chart image
# ----------------------------------------------------------------
tags_pool = [
    "none", "funny", "minecraft", "comedy", "challenge", "rap", "vlog",
    "football", "gaming", "music", "nba", "hip hop", "highlights", "news",
    "sports", "animation", "family friendly", "fortnite", "tiktok", "trailer",
    "video", "new", "gameplay", "how to", "basketball", "family", "soccer",
    "game", "reaction", "diy", "nfl", "boxing", "records", "review", "fun",
    "pop", "games", "science", "entertainment", "live"
]

print("Starting Kafka fake-data producer for regions:", regions)
try:
    while True:
        region = random.choice(regions)
        topic  = f"youtube_{region}"

        record = {
            'video_id'         : str(uuid.uuid4())[:12],
            'title'            : f"Video {random.randint(1,9999)}",
            'publishedAt'      : datetime.utcnow().isoformat() + 'Z',
            'channelId'        : f"channel_{random.randint(1000,9999)}",
            'channelTitle'     : f"Channel {random.choice(['MrBeast','James','Muz','DrLi','DrLeap','ProfReddig','DrWang'])}",
            'categoryTitle'    : random.choice(categories),
            'trending_date'    : datetime.utcnow().strftime('%y.%d.%m'),
            'tags'             : '|'.join(random.sample(tags_pool, k=random.randint(1,3))),
            'view_count'       : random.randint(0,50000),
            'likes'            : random.randint(0,20000),
            'dislikes'         : random.randint(0,5000),
            'comment_count'    : random.randint(0,5000),
            'thumbnail_link'   : 'http://example.com/thumb.jpg',
            'comments_disabled': random.choice([True, False]),
            'ratings_disabled' : random.choice([True, False])
        }

        producer.send(topic, record)
        producer.flush()
        print(f"Sent to {topic} → category:{record['categoryTitle']}, tags:{record['tags']}")
        time.sleep(5)

except KeyboardInterrupt:
    print("\nProducer stopped.")
