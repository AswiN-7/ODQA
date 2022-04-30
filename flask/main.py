

# from ODQA.odqa_answer_extractor import answer_extract

from flask import Flask, render_template, request, redirect, send_from_directory 
app = Flask(__name__)


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/qa', methods= ['GET', 'POST'])
def qa():
    result = None
    if request.method == 'GET':
        return render_template('qa.html', result = result)
    else:
        ques_con = request.form
        context = ques_con['context']
        question = ques_con['question']


        QA_input = {
            "context" : context, 
            "question" : question 
        }
        # res = answer_extract(QA_input)
        # print(res)
        # QA_input['answer'] = res
        return render_template('qa.html', result = QA_input)




if __name__ == '__main__':
    app.debug = True
    app.run()