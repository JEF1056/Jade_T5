import t5
import os
import seqio
import functools
from google.cloud import storage
import tensorflow.compat.v1 as tf

client = storage.Client()

def dataset_fn(split, shuffle_files=False):
    global nq_tsv_path
    del shuffle_files
    # Load lines from the text file as examples.
    files_to_read=[os.path.join("gs://"+nq_tsv_path["bucket"],str(filename.name)) for filename in client.list_blobs(nq_tsv_path["bucket"], prefix=nq_tsv_path[split]) if str(filename.name).startswith(os.path.join(nq_tsv_path[split],f'{nq_tsv_path["taskname"]}-{split}'))]
    #print(len(files_to_read))
    #print(files_to_read[0:10])
    ds = tf.data.TextLineDataset(files_to_read, compression_type=nq_tsv_path["compression"]).filter(lambda line:tf.not_equal(tf.strings.length(line),0))
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.shuffle(buffer_size=600000)
    ds = ds.map(functools.partial(tf.io.decode_csv, record_defaults=["",""], field_delim="\t", use_quote_delim=False),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
    return ds

def preprocess(ds):
    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs": ex["question"],
            "targets": ex["answer"]
        }
    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
    "targets":
        seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


def create_registry(bucket, train, val, taskname, compression_type):
    global nq_tsv_path
    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~registering {taskname}")
    nq_tsv_path={"bucket": bucket, "taskname":taskname, "train":train, "validation":val, "compression": compression_type}
    seqio.TaskRegistry.add(
        taskname,
        # Specify the task source.
        source=seqio.FunctionDataSource(
            # Supply a function which returns a tf.data.Dataset.
            dataset_fn=dataset_fn,
            splits=["train", "validation"]),
        # Supply a list of functions that preprocess the input tf.data.Dataset.
        preprocessors=[
            preprocess,
            seqio.preprocessors.tokenize_and_append_eos,
        ],
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text,
        # We'll use accuracy as our evaluation metric.
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=DEFAULT_OUTPUT_FEATURES,
    )
    
if __name__ == '__main__':
    create_registry("C:/Users/Jess_Fan/Documents/Jade_T5/src/context-val.txt.gz", "C:/Users/Jess_Fan/Documents/Jade_T5/src/context-val.txt.gz", "all_mix", "GZIP")