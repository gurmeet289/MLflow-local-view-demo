# MLflow-local-view-demo

## 📘 Sentiment Analysis with DistilBERT & MLflow Tracking

    This project demonstrates how to fine-tune a pre-trained DistilBERT model on the IMDb movie review dataset using Hugging Face Transformers, and track experiments with MLflow.

## 📂 Project Structure
    .
    ├── train_sentiment_model.py  # Main training script
    ├── distilbert_sentiment_model/  # Saved fine-tuned model
    └── README.md
    
## 🚀 Requirements
    accelerate==1.6.0
    aiohappyeyeballs==2.6.1
    aiohttp==3.11.18
    aiosignal==1.3.2
    alembic==1.15.2
    annotated-types==0.7.0
    anyio==4.9.0
    async-timeout==5.0.1
    attrs==25.3.0
    blinker==1.9.0
    cachetools==5.5.2
    certifi==2025.4.26
    charset-normalizer==3.4.1
    click==8.1.8
    cloudpickle==3.1.1
    colorama==0.4.6
    contourpy==1.3.0
    cycler==0.12.1
    databricks-sdk==0.51.0
    datasets==3.5.1
    Deprecated==1.2.18
    dill==0.3.8
    docker==7.1.0
    exceptiongroup==1.2.2
    fastapi==0.115.12
    filelock==3.18.0
    Flask==3.1.0
    fonttools==4.57.0
    frozenlist==1.6.0
    fsspec==2025.3.0
    gitdb==4.0.12
    GitPython==3.1.44
    google-auth==2.39.0
    graphene==3.4.3
    graphql-core==3.2.6
    graphql-relay==3.2.0
    greenlet==3.2.1
    h11==0.16.0
    huggingface-hub==0.30.2
    idna==3.10
    importlib_metadata==8.6.1
    importlib_resources==6.5.2
    itsdangerous==2.2.0
    Jinja2==3.1.6
    joblib==1.4.2
    kiwisolver==1.4.7
    Mako==1.3.10
    Markdown==3.8
    MarkupSafe==3.0.2
    matplotlib==3.9.4
    mlflow==2.22.0
    mlflow-skinny==2.22.0
    mpmath==1.3.0
    multidict==6.4.3
    multiprocess==0.70.16
    networkx==3.2.1
    numpy==2.0.2
    opentelemetry-api==1.32.1
    opentelemetry-sdk==1.32.1
    opentelemetry-semantic-conventions==0.53b1
    packaging==24.2
    pandas==2.2.3
    pillow==11.2.1
    propcache==0.3.1
    protobuf==6.30.2
    psutil==7.0.0
    pyarrow==19.0.1
    pyasn1==0.6.1
    pyasn1_modules==0.4.2
    pydantic==2.11.4
    pydantic_core==2.33.2
    pyparsing==3.2.3
    python-dateutil==2.9.0.post0
    pytz==2025.2
    pywin32==310
    PyYAML==6.0.2
    regex==2024.11.6
    requests==2.32.3
    rsa==4.9.1
    safetensors==0.5.3
    scikit-learn==1.6.1
    scipy==1.13.1
    six==1.17.0
    smmap==5.0.2
    sniffio==1.3.1
    SQLAlchemy==2.0.40
    sqlparse==0.5.3
    starlette==0.46.2
    sympy==1.14.0
    threadpoolctl==3.6.0
    tokenizers==0.21.1
    torch==2.7.0
    tqdm==4.67.1
    transformers==4.51.3
    typing-inspection==0.4.0
    typing_extensions==4.13.2
    tzdata==2025.2
    urllib3==2.4.0
    uvicorn==0.34.2
    waitress==3.0.2
    Werkzeug==3.1.3
    wrapt==1.17.2
    xxhash==3.5.0
    yarl==1.20.0
    zipp==3.21.0


## Install with:
    pip install torch transformers datasets scikit-learn mlflow

## ▶️ Run the Script

    Save the code in a file called train_sentiment_model.py and run:
    python train_sentiment_model.py

    This will:
    
        1) Load and tokenize IMDb data
        
        2) Fine-tune distilbert-base-uncased for sentiment classification
        
        3) Evaluate accuracy, precision, recall, and F1 score
        
        4) Track all parameters, metrics, and the model using MLflow

## 📊 View MLflow UI Locally
    To launch the MLflow tracking UI in your browser:
    mlflow ui

    This starts a local server at:
      http://127.0.0.1:5000

    You can see:

        1) Parameters (like model name, epochs, batch size)
        
        2) Evaluation metrics
        
        3) Logged artifacts (like the saved model)

## 📁 Output
    Trained model saved in distilbert_sentiment_model/

    MLflow logs experiment data in mlruns/ (auto-created)

## 🧪 Example Evaluation Output (via MLflow)
    Metric	Value
    Accuracy	~0.85
    Precision	~0.86
    Recall	~0.84
    F1 Score	~0.85

    (Numbers will vary depending on system and randomness)

## 📝 Notes

    1) Only 5% of the IMDb dataset is used for quick training (~1-2 mins on CPU).
    
    2) Model is trained for just 1 epoch; increase epochs for better results.
    
    3) Best run with a GPU for faster training.






