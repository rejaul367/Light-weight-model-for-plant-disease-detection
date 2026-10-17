# SMG_Lite — experiment project

One notebook, one small cell per experiment, and a final cell that writes every number and
figure into a filled copy of `result_word.docx`.

## Quick start on Google Colab (fast and reliable, automatic)

**1. Upload once** to Google Drive, exactly like this:
```
MyDrive/SMG_Lite/
├── SMG_Lite_Experiments.ipynb  ← the notebook (open this one in Colab)
├── SMG_Lite_Project.zip        ← this whole project folder, zipped
└── datasets/
    ├── plantvillage.zip        ← zip of the class folders
    ├── rice.zip
    └── mango.zip
```

**2. Open the notebook in Colab.** Double-click `SMG_Lite_Experiments.ipynb` on Drive and pick Open with → Google Colaboratory. Choose Runtime → Change runtime type → GPU, then run the
cells from the top. Run only the experiment cells you want.

**What the notebook does for you:**

| | Where | Why |
|---|---|---|
| Project + images | copied/unzipped to Colab's local disk `/content/` | **fast** training |
| Checkpoints, results, figures, docx | saved locally **and** copied to `MyDrive/SMG_Lite/outputs/` after every epoch | **reliable**: nothing lost if Colab disconnects |
| New session | everything is restored from Drive automatically | finished cells are skipped, an unfinished cell continues from its last epoch |

Your final results are in `MyDrive/SMG_Lite/outputs/`; the filled Word file is
`result_word_filled.docx` in that folder. You do not need to copy anything by hand.

Zip tips: a zip with an extra top folder (`rice.zip → RiceDataset/<classes>`) and Mac
`__MACOSX` folders are handled automatically. Drive free space: about 4–6 GB is used in total.
VGG16 needs about 3 GB extra while it trains; that is freed when its cell finishes.

**On your own PC or Kaggle:** put the images in `data/<dataset>/` and run the notebook. No
Drive is needed.

## Folder structure

```
SMG_Lite_Project/
├── SMG_Lite_Experiments.ipynb   ← the single notebook (run top to bottom)
├── SMG_Lite.py                  ← the model (phytonet.py with only the names changed)
├── smg_lab.py                   ← experiment engine: split, train, test, metrics, figures, docx
├── train.py                     ← original command-line trainer (renamed to SMG_Lite)
├── benchmark.py                 ← original params / FLOPs script (renamed to SMG_Lite)
├── README_model.md              ← original model README (renamed to SMG_Lite)
├── requirements.txt
├── templates/
│   └── result_word.docx         ← your empty template (never modified)
├── data/                        ← put the datasets here (one folder per class)
│   ├── plantvillage/            ←   e.g. plantvillage/color/Apple___Apple_scab/*.jpg
│   ├── rice/                    ←   e.g. rice/Bacterial_leaf_blight/*.jpg
│   └── mango/                   ←   e.g. mango/Anthracnose/*.jpg
└── outputs/                     ← created by the notebook
    ├── splits/<dataset>.json            fixed stratified train / val / test split (70/15/15)
    ├── runs/<dataset>/<model>/          best.pt, metrics.json, history.json, preds.npz
    ├── ablation/<dataset>/G?_M?_C?/     same for the 8 ablation variants
    ├── tables/                          table_<dataset>.csv/.png, ablation_<dataset>.csv/.png
    ├── figures/                         metrics_lines_*, curves_*, confusion_*, roc_*
    └── result_word_filled.docx          ← the template with all tables and figures filled
```

## How the notebook maps to `result_word.docx`

| Notebook section | Cells | Fills |
|---|---|---|
| 0 Setup | config + sanity check | — |
| A Plant Village | 8 models + 2 no-augmentation runs | table 1, and one panel of every figure |
| B Rice | same 10 cells | table 2, one panel of every figure |
| C Mango Leaf | same 10 cells | table 3, one panel of every figure |
| D Ablation | 8 variants (Ghost / MS-DW gate / CEA on-off) | ablation table |
| E Combine | 2 cells | `outputs/result_word_filled.docx` + PNG/CSV |

Each table row gets: model size (MB), trainable parameters, test loss, accuracy, precision,
recall, F1 (macro), AUC (one-vs-rest, macro), Cohen's kappa and GFLOPs. The best value in each
metric column is set in bold. Confusion matrices and ROC curves are for SMG_Lite. The training-curve
figure compares MobileNetV2 and SMG_Lite, with and without augmentation, like the template.

**The ablation** runs on its own dataset (`ABLATION_DATASET`, default `rice`) with its own epoch
budget (`ABLATION_EPOCHS`, default 20). Change either in the config cell. To ablate on a second
dataset, change `ABLATION_DATASET` and run section D again; section E then adds a table for it.
`SMG_Lite.py` is never edited. The components are switched off when the model is built:
* Ghost ✗ → a plain 1×1 expansion
* MS-DW gate ✗ → a single depthwise 3×3
* CEA ✗ → identity

## Data

* **Splits:** 70 / 15 / 15 stratified, seed 42, saved to `outputs/splits/`, so every model
  sees exactly the same images. The model is selected on validation and tested once on test.
  If a dataset folder already has `train/ val/ test/` sub-folders, those are used as-is.
* **Plant Village:** use the original PlantVillage images. If you point the path at the
  folder that contains `color/ grayscale/ segmented/`, the `color` images are used
  automatically. Avoid the Kaggle "New Plant Diseases Dataset (Augmented)": it contains
  offline-augmented copies of the same leaf, which leak between train and test and inflate
  accuracy.
* **Rice / Mango Leaf:** any folder-per-class dataset works; the number of classes is detected.

## Getting high accuracy (applied identically to every model, so the comparison stays fair)

* **Recipe:** the `train.py` recipe (RandAugment, flips, rotation, colour jitter, random
  erasing, label smoothing, MixUp/CutMix, AdamW, warm-up + cosine, EMA, AMP). MixUp/CutMix is
  switched off for the last 20 % of epochs so the model settles on clean images.
* **Checkpoint selection:** every epoch both the raw and the EMA weights are scored on
  validation. The better one is kept, and only that checkpoint is tested.
* **Test-time augmentation:** the original and a horizontally flipped image are averaged
  (`tta=True`).
* **Transfer initialisation:** Rice and Mango runs start from the same model's Plant Village
  weights (`transfer_from="plantvillage"`). This helps most on small datasets. Set it to `None`
  to train every dataset from scratch. Either way, state it in the paper's method section.
* **More epochs is the strongest remaining lever:** 30 → 60–100 on Plant Village. Change
  `epochs` in `DATASETS`.

All models are built from scratch (`weights=None`). The one exception is VGG16, which uses
lr = 1e-4: without BatchNorm it diverges at 3e-3. The numbers you report are whatever these
runs produce on your machine; nothing in the tables is typed in by hand.

## Running

```bash
pip install -r requirements.txt
jupyter notebook SMG_Lite_Experiments.ipynb
```

* **Check the pipeline first:** set `QUICK_TEST = True` (1–2 epochs, 3 batches, results in
  `outputs_quicktest/`). Run all cells, check that the docx gets filled, then set it back to
  `False`.
* **Colab:** see "Quick start on Google Colab" at the top. Drive backup and restore are
  automatic.
* **Interrupted runs:** a cell resumes from its last epoch, and finished cells are skipped.
* **Heavy baselines:** VGG16, ResNet50, InceptionV3, Xception and DenseNet121 take hours on
  Plant Village even on a GPU. Each one is a separate cell, so they can be run in separate
  sessions.
