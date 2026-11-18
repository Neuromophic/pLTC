import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pprint
from pathlib import Path

import torch

import pNN as pNN
from configuration import parser
from utils.checkpoint import CheckpointStore
from utils import (
    CreateUUID,
    CodeFingerprint,
    FormulateArgs,
    GetDataLoader,
    GetMessageLogger,
    MakeFolder,
    SetSeed,
    Trainer,
    TemporalTrainer,
)


def run_experiment(argv=None, *, max_hours=None):
    args = parser.parse_args(argv)
    args = FormulateArgs(args)
    SetSeed(args.SEED)

    print(f"Training network on device: {args.DEVICE}.")
    MakeFolder(args)

    train_loader, datainfo = GetDataLoader(args, "train")
    valid_loader, datainfo = GetDataLoader(args, "valid")
    pprint.pprint(datainfo)

    if args.task == "temporal":
        args.source_sha256 = CodeFingerprint()

    setup = CreateUUID(args, datainfo["dataname"])
    print(f"Training setup: {setup}.")

    msglogger = GetMessageLogger(args, setup)
    msglogger.info(f"Training network on device: {args.DEVICE}.")
    msglogger.info(f"Training setup: {setup}.")
    msglogger.info(datainfo)

    model_path = os.path.join(args.savepath, f"pNN_{setup}")
    if os.path.isfile(model_path):
        checkpoint_store = CheckpointStore(args.temppath, setup)
        if checkpoint_store.is_completed(model_path):
            print(f"Training already completed for UUID: {setup}. Skipping.")
            msglogger.info("Training was already finished.")
            return Path(model_path)
        raise RuntimeError(
            f"Completed artifact exists but is not completed: {model_path}. "
            "Move it aside before starting a new run."
        )

    topology = [datainfo["N_feature"]] + args.hidden + [datainfo["N_class"]]
    msglogger.info(f"Topology of the network: {topology}.")

    if args.task == "temporal":
        result = TemporalTrainer(train_loader, valid_loader, args, setup, topology,
                              model_path, msglogger).fit(max_hours=max_hours)
    else:
        result = train_static(args, topology, train_loader, valid_loader, setup, model_path, msglogger)

    if result.completed:
        msglogger.info("Training is finished.")
        return Path(model_path)
    elif result.status == "timed_out":
        msglogger.warning("Time limit reached; further training is necessary.")
    else:
        raise RuntimeError(f"Unexpected training status: {result.status}.")


def train_static(args, topology, train_loader, valid_loader, setup, model_path, msglogger):
    pnn = pNN.pNN(topology, args).to(args.DEVICE)
    lossfunction = pNN.Lossfunction(args).to(args.DEVICE)
    learnable_params = pnn.GetParam(args.learnable_group)
    optimizer = torch.optim.Adam(learnable_params, lr=args.LR)

    return Trainer(
        model=pnn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lossfunction=lossfunction,
        optimizer=optimizer,
        args=args,
        logger=msglogger,
        run_id=setup,
        topology=topology,
        final_path=model_path,
    ).fit()


if __name__ == "__main__":
    if run_experiment() is None:
        raise SystemExit("Training interrupted. Rerun this command to resume.")
