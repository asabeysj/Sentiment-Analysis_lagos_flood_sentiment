lagos_flood_sentiment/
│
├── app.py                        # Streamlit UI (dashboard only)
├── models.py                     # Loading & prediction logic (SVM + BERT)
├── data_pipeline.py              # Dataset loading, synthetic generation, future real-tweet ingestion
├── config.py                     # Paths, keywords, locations, constants
├── requirements.txt
├── svm_pipeline.pkl
├── bert_lagos_model/             # HF model + tokenizer
├── synthetic_lagos_floods.csv
└── .streamlit/
    └── config.toml               # theme config
