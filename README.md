# pLTC: Printed Neural Classifier with Liquid Time Constants via Gated RC Dynamics

This repository contains the PyTorch circuit models and experiments for the DATE 2027 paper.

Cite as

```text
Haibin Zhao and Wen-Sheng Zhao.
pLTC: Printed Neural Classifier with Liquid Time Constants via Gated RC Dynamics.
Design, Automation & Test in Europe Conference (DATE), 2027.
```

## Usage

Run the commands from the directory containing `experiment.py`. The generator uses only the Python standard library. For training, evaluation, and plotting, use Python 3.9 with the dependencies below. The eight datasets and their training, validation, and test splits are in `dataset/temporal/`. Circuit surrogate models are in `models/assets/`.

```bash
python -m pip install -r requirements.txt
```

**Quick test.** Set `test_mode = True` in `SbatchGenerator.ipynb` and run all cells. This generates one training command and one evaluation command per configuration for FreezerSmallTrain (`--DATASET 0 --SEED 0`). The default test limit is 10 epochs. Each command writes to its `--projectname` directory. If test and full runs use the same project names, use a separate release copy for the quick test so that each dataset and seed identifies one checkpoint.

```bash
bash -e jobs/test/train.txt
bash -e jobs/test/evaluate.txt
```

On Windows, paste individual lines into CMD. To train and evaluate just the first configuration from PowerShell, run these commands in order.

```powershell
cmd.exe /d /c (Get-Content jobs/test/train.txt -TotalCount 1)
cmd.exe /d /c (Get-Content jobs/test/evaluate.txt -TotalCount 1)
```

1. **Train the printed networks.** Set `test_mode = False` (default). Run all cells in `SbatchGenerator.ipynb` or execute the first command below to generate the command lists and Slurm scripts. On the cluster, activate the same Python environment before submitting. Adjust the partition in the generator for your cluster.

   ```bash
   python -m nbconvert --execute --to notebook --stdout SbatchGenerator.ipynb > /dev/null
   sbatch jobs/train.slurm
   ```

   Each line of `jobs/train.txt` is a complete Bash command with explicit dataset, seed, project directory, model, circuit and training arguments. The generator is the only place to edit the experiment matrix. Training and evaluation run independently of the notebook. To run the first generated training command:

   ```bash
   head -n 1 jobs/train.txt | bash -e
   ```

   `--projectname prnn-1` stores outputs under `./prnn-1/`. Grid names give the resistance in MΩ and capacitance in nF. The actual device values are passed with `--gated-config`.

   Dataset IDs 0–7 correspond to FreezerSmallTrain, MoteStrain, TwoLeadECG, PowerCons, CBF, ItalyPowerDemand, BME and Slope, respectively.

   The reference configurations are `prnn-1`, `prnn-2`, and `pltc-ref`. The nine pLTC grid configurations combine three resistances (0.1, 1, and 10 MΩ) with three capacitances (1, 10, and 100 nF). The four remaining variants are `pltc-learned-r`, `pltc-constant-gate-input`, `pltc-rc-only-fixed-r`, and `pltc-rc-only-learned-r`.

   Each list contains 3,840 commands ordered by configuration, dataset ID, and seed (16 configurations × 8 datasets × 30 seeds, numbered 0–29). With `commands_per_job = 16`, Slurm runs 240 array tasks for training and 240 for evaluation.

2. **Find or resume a training run.** Under `<projectname>/`, `models/` contains the completed checkpoints, `log/` contains training logs, and `temp/` contains intermediate checkpoints. Rerun the same command to resume interrupted training or skip completed models. Use the same code revision, datasets, and circuit assets for training and evaluation. Temporal checkpoints store a fingerprint of the Python sources and circuit surrogate files. Even edits to source comments change that fingerprint.

3. **Evaluate the trained models.** After training completes, submit the evaluation array. To evaluate a single model, use the same dataset, seed and project name as in training.

   ```bash
   sbatch jobs/evaluate.slurm
   # Single dataset and seed:
   python3 evaluation.py --DATASET 0 --SEED 0 --projectname prnn-1 --trace-samples 1
   ```

   Each evaluation writes a JSON record to `<projectname>/eval/eval-*/result.txt`. `validation_metrics.accuracy` and `metrics.accuracy` give validation and test accuracy as fractions. `device_count.capacitor` is the capacitor count. The generated commands also save a trace for the first test sample.

4. **Generate figures and tables.** After evaluating the full experiment, run `PLOT.ipynb` with `jobs/train.txt`, the checkpoints, and the evaluation records available. It selects seeds and pLTC configurations by validation accuracy, using validation loss as the first tie-breaker. The gate dynamics figure uses the first MoteStrain test sample.

   | Output in `figures/` | Contents |
   | --- | --- |
   | `accuracy_capacitors.csv` | Test accuracy (%) and capacitor count for pRNN-1, pRNN-2, and pLTC |
   | `selected_models.csv` | Validation-selected configurations, seeds, and model paths |
   | `gate_ablation.csv` | Test accuracy (%) for the full model and ablations |
   | `trainability.csv`, `trainability.pdf` | Test accuracy across the 30 seeds of each selected configuration |
   | `gate_dynamics.pdf` | Controlled RC responses and a trained pLTC state trajectory |

   ```bash
   python -m nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --stdout PLOT.ipynb > /dev/null
   ```
