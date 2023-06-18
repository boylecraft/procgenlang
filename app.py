import random
import ssl

from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
import base64
from typing import List
import numpy as np
from generate import generate_letters, make_letter_png, Letter
from base64 import b64encode

app = Flask(__name__)

@app.route('/api/generate_letters', methods=['POST'])
def api_generate_letters():
    data = request.json
    N = data.get('N', 26)
    i = data.get('i', 32)
    j = data.get('j', 32)
    branch_rate = data.get('branch_rate', 0.5)
    decay_rate = data.get('decay_rate', -0.25)
    start_i = data.get('start_i', 0.5)
    start_j = data.get('start_j', 0.5)
    jitter = data.get('jitter', 0.05)
    prune_threshold = data.get('prune_threshold', 4)
    direction_weight = data.get('direction_weight', 0)
    line_rate = data.get('line_rate', 0.5)
    line_decay = data.get('line_decay', -0.1)
    max_lines = data.get('max_lines', 5)
    color_lines = data.get('color_lines', False)
    width = data.get('width', 1000)
    spacing = data.get('spacing', 10)
    filename = data.get('filename', 'letters.png')
    SEED = data.get('SEED', 1234)

    letters = generate_letters(N, i, j, branch_rate, decay_rate, start_i, start_j, jitter, prune_threshold, direction_weight, line_rate,
                               line_decay, max_lines, color_lines, SEED)

    make_letter_png(letters, spacing, filename, color_lines, width)

    with open(filename, "rb") as image_file:
        encoded_string = b64encode(image_file.read()).decode('utf-8')
    return {'image': encoded_string}

# SSL context setup
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('/etc/letsencrypt/live/games.boylecraft.net/fullchain.pem', '/etc/letsencrypt/live/games.boylecraft.net/privkey.pem')

    # return send_file(filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
