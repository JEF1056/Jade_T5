import json
import requests
import argparse

parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-url', type=str, default="https://woz-model.herokuapp.com/v1/models/jade:predict",
                    help='Link to the server hosting the model')
args = parser.parse_args()

history=[]

while True:
    inp=input('> ')
    response = requests.post(args.url, data='{"inputs": ["Input: '+inp+' Context: ['+'<div>'.join(history)+']"]}')
    message=json.loads(response.text.replace("⁇ ","<"))["outputs"]["outputs"][0]
    print(message.replace("<br>", "\n"))
    history.append(inp)
    history.append(message)
    history=history[-4:]