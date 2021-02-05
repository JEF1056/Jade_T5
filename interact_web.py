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

args.url="https://"+args.url+"/v1/models/jade:predict"

class ResponseGenerator:    
    def __init__(self, url): 
        self.history=[]
        self.url=url
        self.normalize_chars={'Š':'S', 'š':'s', 'Ð':'Dj','Ž':'Z', 'ž':'z', 'À':'A', 'Á':'A', 'Â':'A', 'Ã':'A', 'Ä':'A',
            'Å':'A', 'Æ':'A', 'Ç':'C', 'È':'E', 'É':'E', 'Ê':'E', 'Ë':'E', 'Ì':'I', 'Í':'I', 'Î':'I',
            'Ï':'I', 'Ñ':'N', 'Ń':'N', 'Ò':'O', 'Ó':'O', 'Ô':'O', 'Õ':'O', 'Ö':'O', 'Ø':'O', 'Ù':'U', 'Ú':'U',
            'Û':'U', 'Ü':'U', 'Ý':'Y', 'Þ':'B', 'ß':'Ss','à':'a', 'á':'a', 'â':'a', 'ã':'a', 'ä':'a',
            'å':'a', 'æ':'a', 'ç':'c', 'è':'e', 'é':'e', 'ê':'e', 'ë':'e', 'ì':'i', 'í':'i', 'î':'i',
            'ï':'i', 'ð':'o', 'ñ':'n', 'ń':'n', 'ò':'o', 'ó':'o', 'ô':'o', 'õ':'o', 'ö':'o', 'ø':'o', 'ù':'u',
            'ú':'u', 'û':'u', 'ü':'u', 'ý':'y', 'ý':'y', 'þ':'b', 'ÿ':'y', 'ƒ':'f',
            'ă':'a', 'î':'i', 'â':'a', 'ș':'s', 'ț':'t', 'Ă':'A', 'Î':'I', 'Â':'A', 'Ș':'S', 'Ț':'T',}
        self.alphabets=urlopen("https://raw.githubusercontent.com/JEF1056/clean-discord/master/src/alphabets.txt").read().decode("utf-8").strip().split("\n")
        for alphabet in self.alphabets[1:]:
            alphabet=alphabet
            for ind, char in enumerate(alphabet):
                try:self.normalize_chars[char]=self.alphabets[0][ind]
                except: print(alphabet, len(alphabet), len(self.alphabets[0]));break
        
        self.r1=re.compile(r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)|[\w\-\.]+@(?:[\w-]+\.)+[\w-]{2,4}|(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|```(?:.?)+```|:[^:\s]*(?:::[^:\s]*)*:|(?:\\n)+|(?<=[:.,!?()]) (?=[:.,!?()])|[^a-z1-9.,!@?\s\/\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+', flags=re.DOTALL | re.IGNORECASE)
        self.r2=re.compile(r'[\U00003000\U0000205F\U0000202F\U0000200A\U00002000-\U00002009\U00001680\U000000A0\t]+')
        self.r5=re.compile(r"([\.\'\"@?!a-z])\1{4,}", re.IGNORECASE)
        self.r6=re.compile(r"\s(.+?)\1+\s", re.IGNORECASE)
        self.r8=re.compile(r"([\s!?@\"\'])\1+")
        self.r9=re.compile(r'\s([?.!\"](?:\s|$))')

    def clean(self, text, author=False):
        unique=[i for i in list(set(text)) if i not in self.alphabets[0]] #handle special chars from other langs
        for char in unique: 
            try: text=text.replace(char, self.normalize_chars[char])
            except:pass
        text= re.sub(self.r1, "", text.strip()) #remove urls, emails, code blocks, custom emojis, spaces between punctuation, non-emoji, punctuation, letters, and phone numbers
        text= re.sub(self.r2, " ", text) #handle... interesting spaces
        text= re.sub(self.r5, r"\1\1\1", text) #handle excessive repeats of punctuation, limited to 3
        text= re.sub(self.r6, r" \1 ", text) #handle repeated words
        text= re.sub(self.r8, r"\1",text) #handle excessive spaces or excessive punctuation
        text= re.sub(self.r9, r'\1', text) #handle spaces before punctuation but after text
        text= text.strip().replace("\n","/n") #handle newlines
        text= text.encode("ascii", "ignore").decode() #remove all non-ascii
        text=text.strip() #strip the line
        if author==True: text=text.split(" ")[-1]
        return text

    def response(self, username, inp, debug=False):
        inp= self.clean(username, author=True)+": "+self.clean(inp)
        inpdata = '{"inputs": ["Input: '+'/b'.join(self.history+[inp])+'"]}'
        response = requests.post(self.url.encode("utf-8"), data=inpdata.encode("utf-8"))
        if debug: print(response.text)
        message = json.loads(response.text)
        if "error" in message: 
            print(f"{message}")
            return str(message)
        self.history.append(inp)
        self.history.append("Jade: "+message["outputs"]["outputs"][0])
        self.history = self.history[-10:]
        return message["outputs"]["outputs"][0].replace("/n", "\n")
        
model=ResponseGenerator(args.url)

if __name__ == '__main__':
    history = []
    uname=input('Username: ')

    while True:
        inp = input('> ')
        print(model.response(uname, inp, debug=args.debug))