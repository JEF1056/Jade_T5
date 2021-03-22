import functools
import tensorflow as tf
import tensorflow_datasets as tfds

nq_tsv_path = {'validation': "src/testfile.txt"}

def gen_perms(ts):
    arr=tf.strings.split(ts, sep="\t")
    return arr

def nq_dataset_fn(split, shuffle_files=False):
    # We only have one file for each split.
    del shuffle_files
    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(nq_tsv_path[split])
    # Split each "<question>\t<answer>" example into (question, answer) tuple.
    ds = ds.map(gen_perms, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #ds = ds.unbatch()
    
    ds = ds.map(lambda *ex, ey: dict(zip(["question", "answer"], ex, ey)))
    return ds

print("A few raw validation examples...")
for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
    print(ex)