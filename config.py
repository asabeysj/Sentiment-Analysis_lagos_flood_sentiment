# config.py

MODEL_DIR_BERT = "asabeysj/lagos-flood-bert"
SVM_MODEL_PATH = "./svm_pipeline.pkl"
DATA_PATH = "./synthetic_lagos_floods.csv"

LABELS = ["negative", "neutral", "positive"]

DISTRESS_KEYWORDS = [
    "help", "trapped", "drowning", "die", "emergency", "stuck",
    "no rescue", "need help", "house don flood", "water don full",
]

# In future, you can add:
# TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
