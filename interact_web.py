import json
import requests
import argparse

def s2b(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-url', type=str, default="https://woz-model.herokuapp.com/v1/models/jade:predict",
                    help='Link to the server hosting the model')
parser.add_argument("-debug", type=s2b, nargs='?', const=True, default=False,
                    help="debugging flag")
args = parser.parse_args()

history=[]

while True:
    inp=input('> ')
    response = requests.post(args.url, data='{"inputs": ["Input: '+inp+' Context: ['+'<div>'.join(history)+']"]}')
    message=json.loads(response.text.replace("⁇ ","<"))
    if "error" in message or args.debug: print(f"\n{message}\n")
    elif not "error" in message: 
        print(message["outputs"]["outputs"][0].replace("<br>", "\n"))
        history.append(inp)
        history.append(message["outputs"]["outputs"][0])
        history=history[-4:]