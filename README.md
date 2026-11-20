# Slicer nnUNet

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/1.png" width="800"/>

## Table of contents

* [Introduction](#introduction)
* [Acknowledgments](#acknowledgments)
* [Using the extension](#using-the-extension)
* [Fine-tuned nnFoundation models](#fine-tuned-nnfoundation-models)
* [Validation](#validation)
* [Changelog](#changelog)
* [Expected Weight Folder Structure](#expected-weight-folder-structure)
* [Contributing](#contributing)

## Introduction

<div style="text-align:center">
<img class="center" src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/SlicerNNUnet/Resources/Icons/SlicerNNUnet.png"/>
</div>

This module allows to install and run nnUNet trained models in 3D Slicer with only the training folder.

It streamlines development and integration of new nnUNet models into the 3D Slicer environment.

## Acknowledgments

This module was originally co-financed by the 
<a href="https://orthodontie-ffo.org/">Fédération Française d\'Orthodontie</a> (FFO) as part of the 
<a href="https://github.com/gaudot/SlicerDentalSegmentator/">Dental Segmentator</a> 
developments and the <a href="https://rhu-cosy.com/en/accueil-english/">Cure Overgrowth Syndromes</a> 
(COSY) RHU Project.

The installation steps are based on the work done in the 
<a href="https://github.com/lassoan/SlicerTotalSegmentator/">Slicer Total Segmentator extension</a>.

This extension interfaces 3D Slicer with the 
<a href="https://github.com/MIC-DKFZ/nnUNet">nnUNet library</a>.

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## Using the extension

The nnFoundation changes are on
[`sash9966/SlicerNNUnet`, branch `feature/nnfoundation-support`](https://github.com/sash9966/SlicerNNUnet/tree/feature/nnfoundation-support).
Clone that branch explicitly:

```bash
git clone --branch feature/nnfoundation-support https://github.com/sash9966/SlicerNNUnet.git
```

This development fork retains the original extension/module names and replaces
it in a Slicer installation. Disable or uninstall the original **NNUNet**
extension first; do not load both copies. For local development add the clone's
`SlicerNNUnet` subfolder to **Application Settings → Modules → Additional module
paths**, then restart. Open **Segmentation → nnUNet**.

For the tested macOS CNN/ViT setup, follow [NATIVE_MACOS.md](NATIVE_MACOS.md),
including its Python-console environment block after each Slicer startup.
Cloning the extension alone does not configure that runtime. Model weights are
supplied separately as their unchanged fine-tuned nnU-Net model folders.

A built extension package and extension-catalogue installation have not been
validated; see [VALIDATION.md](VALIDATION.md) for the source-loaded checks.

Once installed, navigate to `Segmentation>nnUNet` in the modules drop down menu or search directly
for `nnUnet`.

Once in the widget, you can install nnUNet's dependencies by clicking on the "nnUNet Install" button.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/2.png"/>

This area will display the current version of nnUNet in 3D Slicer's environment.
If nnUNet is not yet installed, the current version will display None.
It can be installed by clicking on the install Button.

Choose the model-compatible version in the "To install" field before explicitly
clicking Install. The installer retains upstream's protections for Slicer-managed
SimpleITK, requests and PyTorch. **Apply never installs, upgrades or replaces a
runtime.** An installed version number alone does not prove compatibility with a
particular trainer or architecture.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/3.png"/>

Note that the extension may require a 3D Slicer restart before running the inference.

Once the nnUNet is correctly installed, click on the 'nnUNet Run Settings' button to set the path to the Model to use.
This path will be saved for further usage after the first segmentation.

The provided model path should contain the nnUNet 'dataset.json' file.

> [!NOTE]
> To test this extension, you can download the 
> <a href="https://github.com/wasserth/TotalSegmentator/releases">Total Segmentator NNUNet weights</a> 
> and use the `CTChest` Sample. 


> [!WARNING]
> The model weight folder structure should follow the nnUNet expected weight structure for this module to work
> (see the [expected weight folder structure](#expected-weight-folder-structure) section)

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/4.png"/>

Select the volume on which to run the model using the volume input editor. If the selected nnUNet model expects 
multichannel input, multiple input volume selectors will be displayed in the expected order.

Then click on the `Apply` button.

The logs console will display all the information regarding running the input model.

Once the model has finished running, the segmentation will be loaded into 3D Slicer with its associated labels.
It can then be viewed and edited in the `Segment Editor` module.
Use **Save** to save the edited segmentation as `.seg.nrrd`, or save the scene with
the input volume and segmentation. Stop cancels the background process. Failed
or cancelled runs do not import output or remove existing scan/segmentation nodes.

<img src="https://github.com/KitwareMedical/SlicerNNUnet/raw/main/Screenshots/1.png"/>

## Expected weight folder structure

To properly run the nnUNet prediction, the nnUNet weight folder structure should be preserved.

```text
DatasetNNN_Name/
  ExistingTrainer__ExistingPlans__3d_fullres/
    plans.json
    dataset.json
    dataset_fingerprint.json
    fold_0/checkpoint_final.pth
    fold_1/checkpoint_final.pth
    fold_2/checkpoint_final.pth
    fold_3/checkpoint_final.pth
    fold_4/checkpoint_final.pth
    fold_all/checkpoint_final.pth
```

Include only the folds/files actually present. Select the existing trainer/plans/
configuration folder. An unambiguous parent or a fold path is also accepted; a
parent containing multiple models requires a specific model selection. The
extension reads the exact plans identifier/configuration from `plans.json` and
matches that suffix to preserve complete trainer/plans names containing `__`.
Do not flatten or rename folders, edit metadata or retag checkpoints.

**Folds:** `0` selects `fold_0`; `0,1,2,3,4` explicitly ensembles those folds;
`all` selects only native `fold_all`. Empty retains upstream's `0` default.
Mixed, duplicate and invalid selections are rejected. Missing folds/checkpoints
fail without selecting another model. Set the actual checkpoint filename under
Run Settings; empty means `checkpoint_final.pth`. `.pth` and `.safetensors` names
are accepted, provided the installed native nnU-Net supports that format.

If this structure is not preserved, the inference will raise an error.

The `Parameter` class provides an `isValid` method to check if the provided model structure is valid.

Please read [the official nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#3d-u-net-cascade) 
for more information.

## Fine-tuned nnFoundation models

The workflow is **Load scan → select registered model → Apply → review/edit
segmentation**. Manual model paths and the existing input-channel/advanced controls
remain available. There is one native `nnUNetv2_predict` route for standard models,
CNN and ViT; no separate architecture predictor is installed. On macOS the existing
sequential preprocessing/export arguments are retained.

1. Obtain the **fine-tuned segmentation export** from its owner. Public MAE
   pretraining checkpoints are not substitutes. This repository contains no weights
   or download URLs. Keep weights outside the installed extension, for example in
   a persistent `nnUNetWeights` directory.
2. Select **CNN** or **ViT** in Registered model. In Run Settings choose that model's
   unchanged folder, explicit folds/checkpoint, device, step size and TTA setting.
   Click **Save/Update**. These two entries initially have no path.
3. Select the scan and any additional channels in the `dataset.json` order. Apply
   starts inference only. Selecting/saving an entry never downloads or installs.
4. Edit results in Segment Editor and save them. CNN and ViT preferences are saved
   independently in Slicer's QSettings and restored after restarting Slicer. Advanced
   changes are temporary until Save/Update is clicked for that registered entry.
5. After moving weights, update the selected entry's path in Run Settings and click
   Save/Update. The export itself is never modified.

These D091 CT configurations were used in the single-scan local CPU workflow
validation:

| Setting | CNN | ViT |
|---|---|---|
| Dataset folder | `Dataset091_ImageCHDPseudoCombinedV2` | Same |
| Trainer | `PretrainedTrainer_CT_Adam150` | `PretrainedTrainer_CT_ViT_Adam150` |
| Plans | `nnFoundation_cnn_CT_Adam150` | `nnFoundation_vit_CT_Adam150` |
| Configuration | `3d_fullres` | `3d_fullres` |
| Saved architecture identifier | `ResEncL` | `PrimusX` |
| Folds / checkpoint | `all` / `checkpoint_final.pth` | Same |
| Step size / disable TTA | `0.5` / checked | Same |
| Channels / external postprocessing | One CT / none | Same |

Both exports retain their saved Z-score normalization, 1-mm plans spacing and
192³ patch. Their output metadata defines 1 LV-BP, 2 RV-BP, 3 LA, 4 RA, 5 Myo,
6 Aorta and 7 Pulmonary. The extension derives names/identities from each export's
metadata and uses native image geometry; it does not impose this label set on
other models.

**Local macOS runtime:** both unchanged checkpoints pass strict native restoration
and complete Apply, editing and save/reload in Slicer 5.12.3 using pinned official
nnU-Net and batchgenerators source, with existing Slicer binary dependencies.
Original trainer files are
discovered through native `nnUNet_extTrainer`; no trainer aliases, checkpoint
conversion or nnssl installation are needed. The default installed nnunetv2
2.8.1 lacks that support, and the tested 2.9.0 source is not yet a PyPI release.
Follow the [tested macOS source setup](NATIVE_MACOS.md) and consult the
[validation results](VALIDATION.md) before deployment. Installed packages and
Slicer startup settings have not been changed.

Current official nnU-Net provides native pretrained CNN and Primus trainers.
Use the verified implementation for the selected export and preserve its original
trainer identity. [nnFoundation](https://github.com/MIC-DKFZ/nnssl),
[official adaptation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md),
[inference guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/run-inference.md).

Troubleshooting:

- **Missing/ambiguous metadata:** select one unchanged model run containing
  `plans.json`, `dataset.json` and the requested folds. Restore original missing
  files rather than synthesizing replacements.
- **Trainer/architecture import error:** read the subprocess traceback and use
  the export's compatible original runtime. Registration validates configuration,
  not tensor restoration. Preserve checkpoint bytes and source provenance.
- **Wrong channels:** provide every input in metadata order, already aligned to
  the first input. The extension does not reorder or register images.
- **Device or memory failure:** the log retains the subprocess diagnostic. Free
  memory or explicitly select suitable hardware. The extension does not change
  device, patch size, TTA or plans to fit. No GPU memory requirement is certified
  for these exports in Slicer.
- **No result after failure/Stop:** this is intentional; only a normal zero exit
  with the exact current-run output can be imported. Existing scene nodes remain.

## Validation

See [VALIDATION.md](VALIDATION.md) for tested behavior, unrun real-model acceptance
checks and reproducible commands. Fast tests require only Python; Slicer geometry/
widget checks use synthetic data. Neither substitutes for CNN/ViT reference parity.

## Contributing

This project welcomes contributions. It follows Slicer's rules for contributions. 
If you want more information about how you can contribute, please refer to
the [CONTRIBUTING.md file](https://github.com/Slicer/Slicer/blob/main/CONTRIBUTING.md).
