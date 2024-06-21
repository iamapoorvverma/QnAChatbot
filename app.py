import os, datetime
from werkzeug.utils import secure_filename
import pdfbox
import spacy
import codecs
from flask import Flask, render_template, jsonify, request, url_for, redirect
from src.components import QueryProcessor, PassageRetrieval, AnswerExtractor

app = Flask(__name__)
SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
query_processor = QueryProcessor(nlp)
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'books')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        FILE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        app.config['FILE_PATH'] = FILE_PATH
        f.save(app.config['FILE_PATH'])
        p = pdfbox.PDFBox()
        p.extract_text(app.config['FILE_PATH'])
        print(app.config['FILE_PATH'])
    return render_template("qa.html")


@app.route('/answer-question', methods=['POST', 'GET'])
def analyzer():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        query = query_processor.generate_query(question)
        file_path = app.config['FILE_PATH']
        text_file_path = file_path[:-3] + "txt"
        text_file = text_file_path.replace('\\', '\\\\')
        doc = codecs.open(text_file, 'r', 'UTF-8').read()
        passage_retriever.fit(doc)
        passages = passage_retriever.most_similar(question)
        answers = answer_extractor.extract(question, passages)
        return jsonify(answers)
    else:
        return render_template('qa.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
