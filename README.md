# pLTC: Printed Neural Classifier with Liquid Time Constants via Gated RC Dynamics

This repository contains the PyTorch circuit models and experiments for the DATE 2027 paper.

Cite as

```text
Haibin Zhao and Wen-Sheng Zhao.
pLTC: Printed Neural Classifier with Liquid Time Constants via Gated RC Dynamics.
Design, Automation & Test in Europe Conference (DATE), 2027.
```

## Usage

1. Run [SbatchGenerator.ipynb](SbatchGenerator.ipynb) to generate the command lists (`train.txt`, `evaluate.txt`) and Slurm scripts (`train.slurm`, `evaluate.slurm`) in `jobs/`. Each line in `train.txt` specifies one training run (examples abbreviated).

   ```bash
   python3 experiment.py --DATASET 0 --SEED 0 --projectname prnn-1 ...
   python3 experiment.py --DATASET 0 --SEED 1 --projectname prnn-1 ...
   ...
   ```

2. Run the commands in `jobs/train.txt` to train the networks. Each run saves its trained model in `<projectname>/models/`.

3. After training, run the `evaluation.py` commands in `jobs/evaluate.txt`. Each run saves validation accuracy, test accuracy, and device counts to `<projectname>/eval/eval-<id>/result.txt`.

4. Run [PLOT.ipynb](PLOT.ipynb) to generate CSV tables (accuracy, capacitor counts, and gate ablations) and PDF plots (trainability and gate dynamics) in `figures/`.
