import re
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

class ResponseGenerator:    
    def __init__(self, url): 
        self.history=[]
        self.url=url
        self.r1=re.compile(r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)|[\w\-\.]+@(?:[\w-]+\.)+[\w-]{2,4}|(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|```(?:.?)+```|:[^:\s]*(?:::[^:\s]*)*:|(?:\\n)+|(?<=[:.,!?()]) (?=[:.,!?()])|[^a-z1-9.,!@?\"\'\s\/\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+', flags=re.DOTALL | re.IGNORECASE)
        self.r2=re.compile(r'[\U00003000\U0000205F\U0000202F\U0000200A\U00002000-\U00002009\U00001680\U000000A0\t]+')
        self.r5=re.compile(r"([\.\'\"@?!a-z])\1{4,}", re.IGNORECASE)
        self.r6=re.compile(r"\s(.+?)\1+\s", re.IGNORECASE)
        self.r8=re.compile(r"([\s!?@\"\'])\1+")
        self.r9=re.compile(r'\s([?.!\"](?:\s|$))')
        
    def clean(self, text, author=False):
        text= re.sub(self.r1, "", text.strip()) #remove urls, emails, code blocks, custom emojis, spaces between punctuation, non-emoji, punctuation, letters, and phone numbers
        text= re.sub(self.r2, " ", text) #handle... interesting spaces
        text= re.sub(self.r5, r"\1\1\1", text) #handle excessive repeats of punctuation, limited to 3
        text= re.sub(self.r6, r" \1 ", text) #handle repeated words
        text= re.sub(self.r8, r"\1",text) #handle excessive spaces or excessive punctuation
        text= re.sub(self.r9, r'\1', text) #handle spaces before punctuation but after text
        text= text.strip().replace("\n","/n") #handle newlines
        text=text.strip()
        if author==True: text=text.split(" ")[-1]
        return text

    def response(self, username, inp, debug=False):
        inp= self.clean(username, author=True)+": "+self.clean(inp)
        inpdata = '{"inputs": ["Input: '+'/b'.join(self.history+[inp])+'"]}'
        response = requests.post(self.url.encode("utf-8"), data=inpdata.encode("utf-8"))
        message = json.loads(response.text)
        if "error" in message or debug: 
            print(f"{message}")
            if "error" in message: return str(message)
        self.history.append(inp)
        self.history.append("Jade: "+message["outputs"]["outputs"][0])
        self.history = self.history[-8:]
        return message["outputs"]["outputs"][0].replace("/n", "\n")
        
model=ResponseGenerator(args.url)

if __name__ == '__main__':
    history = []
    uname=input('Username: ')

    while True:
        inp = input('> ')
        print(model.response(uname, inp, debug=args.debug))