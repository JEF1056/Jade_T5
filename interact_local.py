import tensorflow as tf
import tensorflow_text  # Required to run exported model.
import argparse
import json
from flask import Flask, request

app = Flask(__name__)

parser = argparse.ArgumentParser(description='Finetune T5')
parser.add_argument('-dir', type=str, help='folder containing the serving model', required=True)
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
        if data==None: data=json.loads(request.text)
        return predict_fn(data["inputs"])[0].decode('utf-8')
        
if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=8051)