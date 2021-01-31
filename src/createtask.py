import t5
import json
import functools
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

with open("config.json", "r") as f:
    nq_tsv_path=json.load(f)

def nq_dataset_fn(split, shuffle_files=False):
    # We only have one file for each split.
    del shuffle_files
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(nq_tsv_path[split])
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.map(functools.partial(tf.io.decode_csv, record_defaults=[''],field_delim="\n", use_quote_delim=False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(ds)
    print(type(ds))
    exit()
    return ds

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
    print(ex)

def preprocess(ds):
    def sample(text):
        print(f"trying {text}")
        text=tf.strings.unicode_encode(text, "UTF-8")
        text=tf.strings.split(text, sep="\t")
        ind=np.sort(np.random.choice(len(text)-1,2, replace=False))
        if ind[1]-ind[0] > 10: ind[0]=ind[1]-10
        return tf.as_tensor('\t'.join(text[ind[0]:ind[1]]), tf.strings.split(text[ind[1]], sep="; ")[1])

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        question, answer = sample(ex)
        return {
            "inputs": tf.strings.join(["Input: ", question]),
            "targets": answer
        }
    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
t5.data.TaskRegistry.add(
    "jade_qa",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[preprocess],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text, 
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy]
)