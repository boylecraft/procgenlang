import requests
import json
import numpy as np
import random

'''
N=18, i=51, j=51, branch_rate=0.99, decay_rate=-0.01, start_i=0.5, start_j=0.5,
                                   jitter=0.1, prune_threshold=5, direction_weight=.80, line_rate=0.99, line_decay=-0.01,
                     max_lines=3, color_lines=False
                     '''

LOCAL_MODE = False

data = {
    'N': 20,
    'i': 51,
    'j': 51,
    'branch_rate': 0.99,
    'decay_rate': -0.01,
    'start_i': 0.5,
    'start_j': 0.5,
    'jitter': 0.1,
    'prune_threshold': 5,
    'direction_weight': .80,
    'line_rate': 0.99,
    'line_decay': -0.01,
    'max_lines': 3,
    'color_lines': False,
    'width': 480,
    'spacing': 1,
    'filename': 'letters.png',
    'SEED' : 1234
}

if LOCAL_MODE:
    url='http://127.0.0.1:5000'
else:
    url='https://games.boylecraft.net:5000'
response = requests.post(f'{url}/api/generate_letters', json=data)

from base64 import b64decode

# after the POST request
image_data = b64decode(response.json()['image'])
with open('letters_api.png', 'wb') as f:
    f.write(image_data)
