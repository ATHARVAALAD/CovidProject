from flask import Flask, render_template, request
import numpy as np
import os
from model import image_pre, predict

app = Flask(__name__)

# Use relative path for better portability
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None  # Default result message

    if request.method == 'POST':
        if 'file1' not in request.files or request.files['file1'].filename == '':
            result = "No file selected! Please upload an image."
        else:
            file1 = request.files['file1']
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
            file1.save(path)

            # Process image and get prediction
            data = image_pre(path)
            if data is None:
                result = "Error processing the image. Try another file."
            else:
                s = predict(data)
                result = "No COVID detected" if s == 1 else "COVID detected"

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
