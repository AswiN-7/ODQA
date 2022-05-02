import ODQA_utils.odqa_milvus
import ODQA_utils.odqa_encoder
import ODQA_utils.odqa_mysql
from ODQA_utils.odqa_answer_extractor import answer_extract

from flask import Flask, render_template, request, redirect, send_from_directory 
import json

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/q', methods= ['GET', 'POST'])
def retrieve_context_and_answer():
    result = None
    if request.method == 'GET':
        return render_template('q.html', result = result)
    else:
        question = request.form['question']
        question_emb = ODQA_utils.odqa_encoder.encode(question)
        similar_ids = ODQA_utils.odqa_milvus.find_similar(question_emb)
        sim_ids = similar_ids[0].ids
        print(sim_ids)
        retrieved_context = ODQA_utils.odqa_mysql.extract_context(sim_ids[0])[0]
        print(retrieved_context)
        QA_input = {
            "context" : retrieved_context, 
            "question" : question 
        }
        res = answer_extract(QA_input)
        print(res)
        QA_input['answer'] = res
        return render_template('q.html', result = QA_input)


@app.route('/qc', methods= ['GET', 'POST'])
def qc():
    result = None
    if request.method == 'GET':
        return render_template('qc.html', result = result)
    else:
        ques_con = request.form
        context = ques_con['context']
        question = ques_con['question']
        QA_input = {
            "context" : context, 
            "question" : question 
        }
        res = answer_extract(QA_input)
        print(res)
        QA_input['answer'] = res
        return render_template('qc.html', result = QA_input)


@app.route('/sampleqa')
def sample_qa():
    fp = r"dataset/tamilqa.json"
    file = open(fp, encoding="utf8")
    qas = json.loads(file.read())
    # print(qas[0])
    return render_template("sampleqa.html", results = qas[:10])

if __name__ == '__main__':
    app.debug = True
    app.run()