import t5
import os
import argparse
import src.createtask
import warnings
import logging as py_logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
py_logging.root.setLevel('INFO')

parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-dir', type=str, required=True,
                    help='Directory of model checkpoints (can/should be a gs:// link)')
parser.add_argument('-out', type=str, default=None,
                    help='Directory to save output')
parser.add_argument('-size', type=str, default="small",
                    help='an integer for the accumulator')
args = parser.parse_args()
parser.add_argument('-tpu_topology', type=str, default="v3-8", choices=["v2-8","v3-8"],
                    help='train file')

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size = {
    "small": (1, 256),
    "base": (2, 128),
    "large": (8, 64),
    "3B": (8, 16),
    "11B": (8, 16)}[args.size]

if args.tpu_topology == "v3-8":
    print("Increasing batches for larger TPU")
    model_parallelism=model_parallelism*4
    train_batch_size=train_batch_size*2

model = t5.models.MtfModel(
    tpu=False,
    model_dir=args.dir,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
)

print("~~Exporting~~")
export_dir = os.path.join(args.dir, "export") if args.out == None else args.out

model.batch_size = 1 # make one prediction per call
saved_model_path = model.export(
    args.out,
    checkpoint_step=-1,  # use most recent
    beam_size=1,  # no beam search
    temperature=0.90,  # sample according to predicted distribution
)
print("Model saved to:", saved_model_path)