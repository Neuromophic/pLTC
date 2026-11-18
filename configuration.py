"""Command-line configuration for pNN experiments."""

import argparse
import json

from utils import ParseBool


parser = argparse.ArgumentParser(
    prog="printed Neural Network",
    description="Experimental setups for printed neural networks.",
)

# Printed circuit parameters
parser.add_argument(
    "--gmin", type=float, default=0.01, help="minimal printable conductance value"
)
parser.add_argument(
    "--gmax", type=float, default=10.0, help="maximal printable conductance value"
)
parser.add_argument("--T", type=float, default=0.1, help="measuring threshold")
parser.add_argument("--m", type=float, default=0.3, help="measuring margin")

# Nonlinear circuits and parameter sharing
parser.add_argument(
    "--activation-circuit",
    type=str.lower,
    choices=("tanh", "hard_sigmoid", "prelu", "sigmoid"),
    default="tanh",
    help="activation circuit type",
)
parser.add_argument(
    "--crossbar-variation",
    type=str.lower,
    choices=("uniform", "lognormal"),
    default="uniform",
    help="crossbar variation model",
)
parser.add_argument(
    "--circuit-variation",
    type=str.lower,
    choices=("uniform", "lognormal"),
    default="uniform",
    help="circuit variation model",
)
parser.add_argument(
    "--circuit-sharing",
    type=str.lower,
    choices=("network", "layer", "neuron"),
    default="network",
    help="learnable circuit parameter sharing scope",
)

# Datasets and data augmentation
parser.add_argument(
    "--task",
    type=str.lower,
    choices=("normal", "split", "temporal"),
    default="normal",
    help="training task type",
)
parser.add_argument("--DATASET", type=int, default=0, help="index of training dataset")
parser.add_argument(
    "--dataset",
    type=str.lower,
    default=None,
    help="dataset name registered in registry.json",
)
parser.add_argument(
    "--DataPath", type=str, default="./dataset", help="path to datasets"
)
parser.add_argument(
    "--dataset-name", type=str, default=None, help="explicit .ds dataset name"
)
parser.add_argument("--gated-config", dest="gated", type=json.loads, default=None,
                    help="explicit version-1 gated configuration as a JSON object")
parser.add_argument("--temporal-model", default=None,
                    help="temporal architecture; omitted preserves the existing pLTC entry")
parser.add_argument("--temporal-protocol", choices=("learnable_filters",), default=None,
                    help="reference stochastic state, crossbar and training semantics; schedule is explicit")
parser.add_argument("--temporal-readout", choices=("last", "sum"), default=None,
                    help="classification readout; reference protocol defaults to summed temporal logits")
parser.add_argument("--initial-state", choices=("constant", "uniform"), default=None,
                    help="uniform U[0,1] for reference protocol; otherwise configured constant voltage")
parser.add_argument("--initial-range", type=float, default=None, metavar="HIGH",
                    help="upper bound of the uniform initial state voltage U[0,HIGH]; default 1")
parser.add_argument("--coupling-factor", type=float, default=None,
                    help="pRNN coupling uniform in [1,1+factor]; reference default 0.1, legacy default 0")
parser.add_argument("--sweep-config", default=None, help="configuration label recorded in sweep results")
parser.add_argument("--train-accuracy-forward", type=ParseBool, default=False,
                    help="recompute training accuracy with a second stochastic forward "
                         "(the published trainer does; only affects the log line)")
parser.add_argument("--gate-lr-scale", type=float, default=1.0,
                    help="learning-rate scale for the pLTC gate crossbar")
parser.add_argument("--target-lr-scale", type=float, default=1.0,
                    help="learning-rate scale for the pLTC target crossbar")
parser.add_argument("--temporal-loss", choices=("last", "mean"), default=None,
                    help="temporal loss over the last input step or mean over all input steps")
parser.add_argument("--recurrent-config", dest="recurrent", type=json.loads, default=None,
                    help="explicit version-1 printed learnable-filter configuration")
parser.add_argument("--temporal-encoding", choices=("stored", "train_minmax"), default=None,
                    help="stored: use saved inputs without scaling/clipping (LearnableFilters); "
                         "train_minmax: legacy train-fitted scaling (default)")
parser.add_argument("--voltage-range", type=float, nargs=2, default=None,
                    metavar=("LOW", "HIGH"),
                    help="encoding range in volts; with stored encoding, reference bounds for reporting only")
parser.add_argument(
    "--batch-size", type=int, default=None, help="batch size; omitted means full-batch"
)
parser.add_argument(
    "--num-workers", type=int, default=0, help="DataLoader worker processes"
)
parser.add_argument(
    "--InputNoise", type=float, default=0.0, help="training input noise"
)
parser.add_argument("--IN_test", type=float, default=0.0, help="test input noise")
parser.add_argument(
    "--R_train", type=int, default=1, help="training input-noise repetitions"
)
parser.add_argument(
    "--R_test", type=int, default=1, help="test input-noise repetitions"
)

# Regularization and network architecture
parser.add_argument(
    "--pathnorm", type=ParseBool, default=False, help="path-norm regularization"
)
parser.add_argument(
    "--hidden",
    type=int,
    nargs="+",
    default=[3],
    help="hidden widths, e.g. --hidden 16 8",
)

# Training
parser.add_argument("--SEED", type=int, default=0, help="random seed")
parser.add_argument("--DEVICE", type=str, default="cpu", help="training device")
parser.add_argument(
    "--PATIENCE",
    type=int,
    default=500,
    help="early-stopping patience for non-progressive training",
)
parser.add_argument("--EPOCH", type=int, default=10**10, help="maximum epochs")
parser.add_argument("--LR", type=float, default=0.1, help="learning rate")
parser.add_argument(
    "--learnable_group", type=str, default="w,n", help="learnable groups: w, n, or w,n"
)
parser.add_argument(
    "--PROGRESSIVE", type=ParseBool, default=True, help="progressive learning rate"
)
parser.add_argument(
    "--LR_PATIENCE", type=int, default=20, help="learning-rate patience"
)
parser.add_argument("--LR_DECAY", type=float, default=0.5, help="learning-rate decay")
parser.add_argument("--LR_MIN", type=float, default=1e-4, help="minimum learning rate")

# Metrics, loss, and time limit
parser.add_argument(
    "--metric",
    type=str.lower,
    choices=("acc", "maa"),
    default="acc",
    help="evaluation metric",
)
parser.add_argument(
    "--FittingLoss",
    type=str.lower,
    choices=("ce", "standard"),
    default="ce",
    help="data-fitting loss",
)
parser.add_argument(
    "--TIMELIMITATION", type=float, default=45, help="maximum runtime in hours"
)

# Device aging and process variation
parser.add_argument(
    "--MODE",
    type=str.lower,
    choices=("nominal", "aging"),
    default="nominal",
    help="training mode",
)
parser.add_argument("--M_train", type=int, default=1, help="training aging models")
parser.add_argument("--K_train", type=int, default=1, help="training temporal samples")
parser.add_argument("--M_test", type=int, default=1, help="test aging models")
parser.add_argument("--K_test", type=int, default=1, help="test temporal samples")
parser.add_argument("--t_test_max", type=int, default=1, help="maximum test interval")
parser.add_argument(
    "--integration",
    type=str.lower,
    choices=("mc", "gq"),
    default="mc",
    help="integration method",
)
parser.add_argument("--N_train", type=int, default=5, help="training variation samples")
parser.add_argument(
    "--e_train", type=float, default=0.1, help="training variation amplitude"
)
parser.add_argument("--N_test", type=int, default=10, help="test variation samples")
parser.add_argument(
    "--e_test", type=float, default=0.2, help="test variation amplitude"
)

# Power estimation and output paths
parser.add_argument(
    "--powerestimator",
    type=str.lower,
    choices=("none", "attraction", "power", "both"),
    default="none",
    help="power regularization type",
)
parser.add_argument(
    "--powerbalance", type=float, default=0.001, help="power regularization scale"
)
parser.add_argument(
    "--estimatorbalance", type=float, default=0.001, help="power estimator balance"
)
parser.add_argument(
    "--pgmin", type=float, default=1e-7, help="minimum physical conductance"
)
parser.add_argument("--projectname", type=str, default="project", help="project name")
parser.add_argument(
    "--temppath", type=str, default="/temp", help="temporary checkpoint path"
)
parser.add_argument("--logfilepath", type=str, default="/log", help="log path")
parser.add_argument(
    "--recording", type=ParseBool, default=False, help="save every epoch"
)
parser.add_argument(
    "--recordpath", type=str, default="/record", help="record checkpoint path"
)
parser.add_argument(
    "--savepath", type=str, default="/experiment", help="completed model path"
)
parser.add_argument(
    "--loglevel",
    type=str.lower,
    choices=("debug", "info"),
    default="info",
    help="log level",
)
