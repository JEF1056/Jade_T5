import t5
import os
import json
import warnings
import argparse
import src.helpers as helpers
import tensorflow.compat.v1 as tf
from contextlib import contextmanager
import logging as py_logging
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Finetune T5')
parser.add_argument('-dir', type=str, default="gs://conversation-t5",
                    help='link to google storage bucket')
parser.add_argument('-train', type=str, default="context-train.txt",
                    help='train file')
parser.add_argument('-val', type=str, default="context-val.txt",
                    help='val file')
parser.add_argument('-tpu_address', type=str, default=None,
                    help='TPU ip address')
parser.add_argument('-tpu_topology', type=str, default="v3-8", choices=["v2-8","v3-8"],
                    help='train file')
parser.add_argument('-in_len', type=int, default=2048,
                    help='train file')
parser.add_argument('-out_len', type=int, default=512,
                    help='train file')
parser.add_argument('-steps', type=int, default=50000,
                    help='train file')
parser.add_argument('-model_size', type=str, default="small", choices=["small", "base", "large", "3B", "11B"],
                    help='train file')
parser.add_argument("-eval", type=helpers.str2bool, nargs='?', const=True, default=False,
                    help="eval model after training")
parser.add_argument('-taskname', type=str, default="jade-qa",
                    help='name of the task')
parser.add_argument('-path', type=str, default="jade-qa",
                    help='name of the task')
args = parser.parse_args()

with open("config.json", "w") as f:
    json.dump({"train":os.path.join(args.dir,"data", args.train), "validation": os.path.join(args.dir,"data", args.val), "taskname":args.taskname},f)
import src.createtask
args.tpu_address = f"grpc://{args.tpu_address}:8470"

if args.tpu_address != None:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_address)
    tf.enable_eager_execution()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    tf.disable_v2_behavior()
    
tf.get_logger().propagate = False
py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
    og_level = tf.logging.get_verbosity()
    tf.logging.set_verbosity(level)
    yield
    tf.logging.set_verbosity(og_level)

MODEL_SIZE = args.model_size
MODELS_DIR = os.path.join(args.dir, "models")
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

if args.tpu_topology == "v3-8":
    print("Increasing batches for larger TPU")
    model_parallelism=model_parallelism*4
    train_batch_size=train_batch_size*2

tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu=args.tpu_address,
    tpu_topology=args.tpu_topology,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    sequence_length={"inputs": args.in_len, "targets": args.out_len},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=2500,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=500,
)

model.finetune(
    mixture_or_task_name=args.taskname,
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=args.steps
)

if args.eval:
    model.batch_size = train_batch_size * 4
    model.eval(
        mixture_or_task_name=args.taskname,
        checkpoint_steps="all"
    )
    
if args.export:
    export_dir = os.path.join(MODEL_DIR, "export")

    model.batch_size = 1 # make one prediction per call
    saved_model_path = model.export(
        args.out,
        checkpoint_step=-1,  # use most recent
        beam_size=1,  # no beam search
        temperature=0.90,  # sample according to predicted distribution
    )
    print("Model saved to:", saved_model_path)