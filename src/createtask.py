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
    #ds.shard(8, 8)
    #ds.shuffle(buffer_size=10000)
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.map(
    functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                      field_delim="\t", use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
    return ds

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
    print(ex)

def preprocess(ds):
    def normalize_text(text):
        #print(f"trying {text}")
        #text=tf.strings.unicode_encode(text, "UTF-8")
        return text

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs":
                tf.strings.join(
                    ["Input: ", normalize_text(ex["question"])]),
            "targets": normalize_text(ex["answer"])
        }
    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
t5.data.TaskRegistry.add(
    nq_tsv_path["taskname"],
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