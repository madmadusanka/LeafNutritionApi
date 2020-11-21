from flask import Flask
from flask import request

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, Farmers!"


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('as.jpg')
        return "work"
    else:
        return "Uploads not work"

app.run(host='127.0.0.1', port=80)