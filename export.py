import t5
import t5.models
import os
import argparse
import warnings
import logging as py_logging
from src.createtask import create_registry
create_registry(None, "src/temp.txt", "src/temp.txt", "all_mix", None, "local")
warnings.filterwarnings("ignore", category=DeprecationWarning)
py_logging.root.setLevel('INFO')

parser = argparse.ArgumentParser(description='Export checkpoints for serving')
parser.add_argument('-dir', type=str, required=True,
                    help='Directory of model checkpoints (can/should be a gs:// link)')
parser.add_argument('-out', type=str, default=None,
                    help='Directory to save output')
parser.add_argument('-name', type=str, default=None,
                    help='Directory to save output')
parser.add_argument('-temperature', type=float, default=0.9,
                    help='model temperature')
parser.add_argument('-beams', type=int, default=1,
                    help='number of beams to use')
parser.add_argument('-batch_size', type=int, default=1,
                    help='model batch size')
args = parser.parse_args()

model = t5.models.MtfModel(
    tpu=False,
    model_dir=args.dir,
    model_parallelism=1,
    batch_size=256,
)

print("~~Exporting~~")
export_dir = os.path.join(args.dir, "export") if args.out == None else args.out

model.batch_size = args.batch_size # make one prediction per call
saved_model_path = model.export(
    args.out,
    checkpoint_step=-1,  # use most recent
    beam_size=args.beams,  # no beam search
    temperature=args.temperature,  # sample according to predicted distribution
)

os.rename(saved_model_path, os.path.join(args.out, args.name))

print("Model saved to:", os.path.join(args.out, args.name))