import tensorflow as tf
import tensorflow_text  # Required to run exported model.
import argparse
import json
from flask import Flask, request
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser(description='Finetune T5')
parser.add_argument('-dir', type=str, help='folder containing the serving model', required=True)
parser.add_argument('-ip', type=str, help='ip address to start the model server on', default="0.0.0.0")
parser.add_argument('-port', type=int, help='port to start the model server on', default=8051)
args = parser.parse_args()

def load_predict_fn(model_path):
  if tf.executing_eagerly():
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures['serving_default'](tf.constant(x))['outputs'].numpy()
  else:
    print("Loading SavedModel in tf 1.x graph mode.")
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    meta_graph_def = tf.compat.v1.saved_model.load(sess, ["serve"], model_path)
    signature_def = meta_graph_def.signature_def["serving_default"]
    return lambda x: sess.run(
        fetches=signature_def.outputs["outputs"].name, 
        feed_dict={signature_def.inputs["input"].name: x}
    )

predict_fn = load_predict_fn(args.dir)

@app.route("/", methods = ['POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        if data==None: data=json.loads(request.data)
        t1=time.time()
        ret_data=predict_fn(data["inputs"])[0].decode('utf-8')
        return {"output":ret_data, "timedelta": str(time.time()-t1)}
        
if __name__ == "__main__":
    app.run(threaded=True, host=args.ip, port=args.port)