import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from scripts.main import ip_main
from scripts.question_to_link import qtl_main
from scripts.links_agregator import la_main


app = Flask(__name__, static_url_path='')


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('test.html')


@app.route('/1', methods=['GET', 'POST'])
def process():
    print('started')
    data = 'None'
    if request.method == 'POST':
        file = request.files['file']
        data = searcher(file)
    return data


def searcher(f):
    pil_image = Image.open(f).convert('RGB')
    open_cv_image = np.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    question = ip_main(open_cv_image)
    link = qtl_main(question)
    text = la_main(link)
    return {'question': question, 'text': text}


if __name__ == '__main__':
    app.run()