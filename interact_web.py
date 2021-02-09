import re
import json
import requests
import argparse
from urllib.request import urlopen

def s2b(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-url', type=str, default="woz-model.herokuapp.com",
                    help='Link to the server hosting the model')
parser.add_argument("-debug", type=s2b, nargs='?', const=True, default=False,
                    help="debugging flag")
args = parser.parse_args()

args.url="http://interact.jadeai.ml/query"

if __name__ == '__main__':
    history = []
    uname=input('Username: ')

    while True:
        inp = input('> ')
        inpdata = '{"inputs": ["Input: '+'/b'.join(["Hi", "hello"])+'"]}'
        response = requests.post(args.url, data=inpdata.encode("utf-8"))
        print(response)