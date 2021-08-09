import t5
import t5.models
import os
import seqio
import warnings
import argparse
import tensorflow.compat.v1 as tf
from contextlib import contextmanager
import logging as py_logging
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Finetune T5')
parser.add_argument('-dir', type=str, default="conversation-t5",
                    help='link to google storage bucket')
parser.add_argument('-tasknames', nargs='+', type=str, default="all-mix",
                    help='name of the task')
parser.add_argument('-gpus', nargs='+', type=str, default=None,
                    help='available GPUs')
parser.add_argument('-train', nargs='+', type=str,  default="context-train.txt",
                    help='train file')
parser.add_argument('-val', nargs='+', type=str, default="context-val.txt",
                    help='val file')
parser.add_argument('-tpu_address', type=str, default=None,
                    help='TPU ip address')
parser.add_argument('-tpu_topology', type=str, default=None, choices=["v2-8","v3-8", None],
                    help='train file')
parser.add_argument('-in_len', type=int, default=2048,
                    help='train file')
parser.add_argument('-out_len', type=int, default=512,
                    help='train file')
parser.add_argument('-steps', type=int, default=50000,
                    help='train file')
parser.add_argument('-model_size', type=str, default="small", choices=["small", "t5.1.1.small", "base", "large", "3B", "11B"],
                    help='train file')
parser.add_argument('-compression', type=str, default=None, choices=[None, "ZLIB", "GZIP"],
                    help='compression the dataset is compressed with')
parser.add_argument('-batch_size', type=int, default=None,
                    help='number of batches')
parser.add_argument('-max_checkpoints', type=int, default=None,
                    help='maximum number of checkpoints')
parser.add_argument('-storemode', type=str, default="gs", choices=["gs", "local"],
                    help='storemode')
parser.add_argument('-model_paralellism', type=int, default=None,
                    help='model_paralellism')
args = parser.parse_args()

from src.createtask import create_registry
for index, name in enumerate(args.tasknames):
    create_registry(args.dir, args.train[index], args.val[index], name, args.compression, args.storemode)

seqio.MixtureRegistry.add(
    "all_mix",
    args.tasknames,
    default_rate=1.0
)

if args.tpu_address != None:
    args.tpu_address = f"grpc://{args.tpu_address}:8470"
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
if args.storemode=="gs": 
    MODELS_DIR = os.path.join("gs://"+args.dir, "models")
    # Public GCS path for T5 pre-trained model checkpoints
    BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
else: 
    MODELS_DIR = os.path.join(args.dir, "models")
    # Public GCS path for T5 pre-trained model checkpoints
    BASE_PRETRAINED_DIR = "pretrained_models"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
try:
    model_parallelism, train_batch_size, keep_checkpoint_max = {
        "small": (1, 512, 4),
        "t5.1.1.small": (1, 512, 4),
        "base": (2, 256, 2),
        "large": (4, 128, 2),
        "3B": (8, 16, 1),
        "11B": (8, 4, 1)}[MODEL_SIZE]
except:
    model_parallelism, train_batch_size, keep_checkpoint_max=None,None,None
    assert args.model_paralellism and args.batch_size and args.max_checkpoints, "Model not found in supported list. Cannot determine model paralellism, batch size, and number of checkpoints to keep automatically."
if args.paralellism: model_paralellism=args.model_paralellism
if args.batch_size: train_batch_size=args.batch_size
if args.max_checkpoints: keep_checkpoint_max=args.max_checkpoints

tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
if args.tpu_address:
    model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
        tpu=args.tpu_address,
        tpu_topology=args.tpu_topology,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={"inputs": args.in_len, "targets": args.out_len},
        learning_rate_schedule=0.001,
        save_checkpoints_steps=2500,
        keep_checkpoint_max=keep_checkpoint_max,
        iterations_per_loop=500,
    )
elif args.gpus:
    model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
        mesh_devices=args.gpus,
        mesh_shape=f'model:1,batch:{len(args.gpus)}',
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={"inputs": args.in_len, "targets": args.out_len},
        learning_rate_schedule=0.001,
        save_checkpoints_steps=2500,
        keep_checkpoint_max=keep_checkpoint_max,
        iterations_per_loop=500,
    )
else: raise NotImplementedError("Running with no accelerators is not a supported case.")
        
model.finetune(
    mixture_or_task_name='all_mix',
    pretrained_model_dir=PRETRAINED_DIR,
    finetune_steps=args.steps
)