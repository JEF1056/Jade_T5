import t5
import functools
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

def dataset_fn(split, shuffle_files=False):
    global nq_tsv_path
    del shuffle_files
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(nq_tsv_path[split], compression_type=nq_tsv_path["compression"]).filter(lambda line:tf.not_equal(tf.strings.length(line),0))
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.shuffle(buffer_size=10000)
    ds = ds.map(functools.partial(tf.io.decode_csv, record_defaults=["",""], field_delim="\t", use_quote_delim=False),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
    return ds

def preprocess(ds):
    def normalize_text(text):
        #print(f"trying {text}")
        #text=tf.strings.unicode_encode(text, "UTF-8")
        return text

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs": normalize_text(ex["question"]),
            "targets": normalize_text(ex["answer"])
        }
    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def create_registry(train, val, taskname, compression_type):
    global nq_tsv_path
    nq_tsv_path={"train":train, "validation":val, "compression": compression_type}
    print("A few raw validation examples...")
    for ex in tfds.as_numpy(dataset_fn("validation").take(5)):
        print(ex)
    t5.data.TaskRegistry.add(
        taskname,
        # Specify the task type.
        t5.data.Task,
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=dataset_fn,
        splits=["train", "validation"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=[preprocess],
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text, 
        # We'll use accuracy as our evaluation metric.
        metric_fns=[t5.evaluation.metrics.accuracy]
    )
    
if __name__ == '__main__':
    create_registry("C:/Users/Jess_Fan/Documents/Jade_T5/src/context-val.txt.gz", "C:/Users/Jess_Fan/Documents/Jade_T5/src/context-val.txt.gz", "all_mix", "GZIP")