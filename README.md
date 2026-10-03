# Toxic Comment Classifier App

A Streamlit application that classifies user-entered text and image captions using a trained LSTM model. Image captions are generated with BLIP-1, and every successful submission is recorded in a SQLite database with its date, time, and predicted labels.

This project was developed as part of my NLP internship.

## Features

- Classify text into six toxicity categories.
- Upload a JPG, JPEG, or PNG image and generate a caption.
- Classify the generated caption using the same LSTM model.
- Display predicted labels and their scores.
- Save text submissions and image captions to SQLite.
- Record repeated submissions as separate rows.

## Classification labels

The model performs **multi-label classification**, so one input may receive multiple labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The LSTM was trained using the Jigsaw Toxic Comment Classification dataset.

## How it works

**Text input:** The text is lowercased and tokenized using NLTK. Tokens are converted to IDs using the vocabulary saved during training, then passed to the LSTM to obtain predictions.

**Image input:** BLIP-1 generates a text caption for the uploaded image. The caption is passed through the text-classification pipeline and saved with its predicted labels.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Streamlit interface and application workflow |
| `textclassification.py` | Text preprocessing, LSTM loading, and classification |
| `imagecaption.py` | BLIP image-caption generation |
| `database.py` | SQLite database setup and submission storage |
| `lstm_Final.pth` | Trained LSTM weights |
| `vocab.txt` | Training vocabulary stored in JSON format |
| `comments.db` | SQLite submission database |
| `quantization_research.ipynb` | Task 0 research notebook |
| `README.md` | Project documentation |

## Setup and usage

Download or clone this repository, then open a terminal in the project folder.

### 1. Install the application dependencies

```bash
python -m pip install torch streamlit transformers pillow nltk
```

SQLite support is included in Python through the `sqlite3` module.

### 2. Download the NLTK tokenizer resources

```bash
python -m nltk.downloader punkt punkt_tab
```

### 3. Check the model files

Ensure `lstm_Final.pth` and `vocab.txt` are available in the project folder. The vocabulary must be the same mapping used to train the saved model.

### 4. Initialize the database if needed

If `comments.db` is not included, run the database setup script once:

```bash
python database.py
```

### 5. Start the application

```bash
python -m streamlit run main.py
```

Open the local URL shown in the terminal. Enter text or upload an image, then submit it to view the results.

BLIP weights are downloaded from Hugging Face on first use, so an internet connection is needed initially. The download may take several minutes; downloaded files are cached for reuse.

## Database records

Submissions are stored in the `comments` table:

| Column | Stored information |
| --- | --- |
| `id` | Unique primary key |
| `text` | Submitted text or generated image caption |
| `date` | Submission date |
| `time` | Submission time |
| `predicted_labels` | Predicted toxicity labels |

Each successful submission creates a new row, including identical comments submitted again. Dates and times use the computer's local time.

The database can be viewed using [DB Browser for SQLite](https://sqlitebrowser.org/).

## Task 0: Quantization research

The research notebook discusses reducing the memory requirements of large models such as BERT and LLaMA through lower-precision representations and quantization.

Open `quantization_research.ipynb` in Jupyter Notebook or VS Code. Keep any referenced image files in their original relative locations.

The Streamlit application is implemented in separate Python files; the notebook is for the research task.

## Limitations

Predictions may be incorrect, especially for text that differs from the training data. For images, the classification applies to the **generated caption** and depends on what BLIP describes. Harmful content that is missing from the caption may therefore be missed.

## References

- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- [BLIP image-captioning base model](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [PyTorch](https://pytorch.org/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

