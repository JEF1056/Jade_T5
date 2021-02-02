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

history = []
username=input('Username: ')

while True:
    inp = input('> ')
    inp= username+": "+inp
    inpdata = '{"inputs": ["Input: '+'\\b'.join(history+[inp])+'"]}'
    response = requests.post(args.url.encode("utf-8"), data=inpdata.encode("utf-8"))
    message = json.loads(response.text)
    if "error" in message or args.debug: 
        print(f"\n{message}\n")
    if not "error" in message: 
        print(message["outputs"]["outputs"][0].replace("\\n", "\n"))
        history.append(inp)
        history.append("Jade: "+message["outputs"]["outputs"][0])
        history = history[-10:]