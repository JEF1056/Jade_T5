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
uname=input('Username: ')

class ResponseGenerator:
    def __init__(self, url): 
        self.history=[""]
        self.url=url

    def response(self, username, inp, debug=False):
        inp= username+": "+inp
        inpdata = '{"inputs": ["Input: '+'/b'.join(self.history+[inp])+'"]}'
        response = requests.post(self.url.encode("utf-8"), data=inpdata.encode("utf-8"))
        message = json.loads(response.text)
        if "error" in message or debug: 
            print(f"{message}")
        if not "error" in message: 
            #print(message["outputs"]["outputs"][0].replace("/n", "\n"))
            self.history.append(inp)
            self.history.append("Jade: "+message["outputs"]["outputs"][0])
            self.history = self.history[-20:]
        return message["outputs"]["outputs"][0].replace("/n", "\n")
        
model=ResponseGenerator(args.url)

while True:
    inp = input('> ')
    print(model.response(uname, inp, debug=args.debug))