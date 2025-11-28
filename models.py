# models.py
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_DIR_BERT, SVM_MODEL_PATH, LABELS
import logging

logger = logging.getLogger(__name__)

_svm_model = None
_tokenizer = None
_bert_model = None


def load_svm_model():
    global _svm_model
    if _svm_model is None:
        logger.info("Loading SVM model from %s", SVM_MODEL_PATH)
        _svm_model = joblib.load(SVM_MODEL_PATH)
    return _svm_model


def load_bert_model():
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        logger.info("Loading BERT model from %s", MODEL_DIR_BERT)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_BERT)
        _bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_BERT)
    return _tokenizer, _bert_model


def predict_svm(text: str) -> str:
    model = load_svm_model()
    try:
        return model.predict([text])[0]
    except Exception as e:
        logger.exception("Error during SVM prediction: %s", e)
        return "neutral"  # safe fallback


def predict_bert(text: str) -> str:
    tokenizer, model = load_bert_model()
    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
        return LABELS[pred_id]
    except Exception as e:
        logger.exception("Error during BERT prediction: %s", e)
        return "neutral"
