# nnFoundation extension validation

Upstream: KitwareMedical/SlicerNNUnet
`45557d4a6688e9802ea3ba8abba46dce91a9e430`.
Local feature branch: `feature/nnfoundation-support`. No GitHub fork or push.

This change affects inference configuration, saved UI preferences and result
handling. No training method, cohort, checkpoint, source export or plans were
modified. Original module names, installer source, license and attribution remain.

## Implemented

- Native numbered folds and `all`, with no fold/checkpoint substitution.
- Full trainer/plans/configuration identities resolved from saved plans metadata;
  no ambiguous `__` splitting or first-match model/CLI selection.
- Exact selected `.pth`/`.safetensors` filename validation.
- Independent CNN/ViT preferences in QSettings; Manual remains available.
- Apply uses the installed runtime without invoking the installer.
- Normal zero process exit and exact fresh output required before import.
  Stop/failure invalidates output; generation guards handle Stop/Apply during UI
  progress callbacks. Diagnostics retain subprocess errors and memory failures.
- Editable segmentation import preserves label identities/names, reference
  geometry and the input's parent transform. Parent transform is assigned before
  the reference geometry so Slicer expresses that geometry in the correct frame.

## Checks and evidence

Checks performed on 17 November 2026. Local Slicer: 5.12.3, Python 3.12.10,
macOS arm64. This is a development test environment; the colleague release target
has not been specified.

Final result: **57 fast tests passed** (28 Parameter, 24 process/import safety,
5 registry); **23 actual Slicer tests passed** (5 installer metadata, 4 real
QProcess, 4 synthetic geometry/scene safety, 10 widget). The two actual-model
parity tests were collected with absent prerequisites and explicitly skipped.

| Check | Status | Evidence / limit |
|---|---|---|
| Standard configuration, fold `0`, explicit ensemble, native `all` | PASS | 28 standard-Python configuration tests |
| Missing folds/checkpoints, invalid/mixed folds, `__` identifiers, spaced paths, malformed metadata | PASS | Same configuration suite |
| Independent CNN/ViT/Manual settings, restart reconstruction, explicit path update | PASS | Standard-Python registry suite and real Slicer widget tests |
| Process errors, crash, cancellation, stale output and reentrant startup | PASS | Fake-process regression suite; actual widget cancellation checks |
| Real Qt subprocess success, memory/dependency error and cancellation | PASS | Four actual QProcess tests in Slicer, including its one-argument finished-signal binding |
| Apply does not install or change dependencies | PASS | Actual Slicer widget tests; InstallLogic.py unchanged |
| Oblique/anisotropic geometry and parent transform | PASS | Actual Slicer NIfTI export/read, segmentation import and labelmap world-affine check; affine tolerance 1e-5 |
| Segment names and sparse IDs | PASS | Actual Slicer IDs 1/7, metadata names and exact per-segment voxel masks |
| Source scan and existing segmentation preserved on failure/Stop | PASS | Actual Slicer synthetic scene tests |
| Both supplied exports' metadata and native CLI arguments | PASS | Actual Slicer Parameter.isValid/asArgList, native `all`, exact original identifiers/final checkpoint, TTA disabled, step 0.5; no tensor load |
| Original legacy trainer in installed Slicer runtime | FAIL | Read-only import raises `ModuleNotFoundError: nnunetv2.training.nnUNetTrainer.pretraining` |
| Actual CNN/ViT native tensor restoration and reference parity | UNRUN | Compatible legacy runtime unavailable locally; no checkpoint loading or prediction executed |
| Standard nnU-Net actual weighted inference | UNRUN | No standard-model reference pair selected for this task; configuration and synthetic I/O checks passed |
| GPU errors on real hardware, memory requirements | UNRUN | Device/diagnostic behavior tested with fakes; no real GPU inference |
| Full install → real CNN/ViT Apply → edit/save in target colleague environment | UNRUN | Target unspecified and legacy runtime unresolved; no extension package built/installed |

Synthetic/fake tests establish configuration and I/O behavior only. Historical
saved model predictions were not recomputed or compared. No numerical tolerance
for real prediction differences is justified by this work. The opt-in parity
tests initially require identical label voxels and matching affines within 1e-5;
any difference requires investigation before approving a tolerance.

## Remaining runtime requirement

Both available D091 exports use original `PretrainedTrainer_CT_Adam150` /
`PretrainedTrainer_CT_ViT_Adam150` trainers and legacy `ResEncL` / `PrimusX`
architecture identifiers. Their shared original trainer code imports an older
pretraining hierarchy absent from local nnunetv2 2.8.1.

Their historical reference nnUNet commit
`202f6baa0adc2ef5f7b615df19cc4da970412cc0` also has a native builder-signature
gap: the CNN builder requires `input_patch_size`, while that native predictor
does not supply it; Primus receives no saved patch size. The already established
original inference implementation supplies the saved patch size explicitly.
Installing the historical package alone therefore does not certify native
prediction compatibility.

Current official nnU-Net has native CNN/Primus support with updated builder
signatures. It is not evidence that these unchanged legacy exports load in a
different runtime. No new trainer/adapter distribution scheme was introduced.
Use the existing compatible implementation and validate a native restoration in
the intended runtime before colleague deployment. **Recommendation: retain this
local implementation; do not describe either real model as Slicer-validated.**

Official documentation/source reviewed:

- [nnFoundation / nnssl](https://github.com/MIC-DKFZ/nnssl), live HEAD
  `3d774e942256bd70e5e264c6b94fa38e25f7ed74`.
- [Official adaptation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md).
- [Official inference guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/run-inference.md).
- [Pinned current predictor](https://github.com/MIC-DKFZ/nnUNet/blob/131289433bdac0610f8b80a0b406e18bcfe51a2a/nnunetv2/inference/predict_from_raw_data.py).
  Standard prediction retains a macOS `0/0` sequential branch which the current
  model-folder entry point lacks; this fork preserves the standard route.

## Reproduce configuration checks

From this checkout:

```bash
python3 -B -m unittest discover -s SlicerNNUnet/Testing/Python -p 'test_*.py' -v
python3 -B -m unittest discover -s SlicerNNUnet/Testing -p 'test_*.py' -v
git diff --check
```

Run the synthetic widget/geometry and installer-metadata checks in Slicer without
user settings, startup scripts, downloads or package changes:

```bash
Slicer --testing --no-splash --no-main-window \
  --python-script /absolute/path/to/SlicerNNUnet/SlicerNNUnet/Testing/run_configuration_checks.py
```

Tests requiring actual weights are separate and opt-in; see
`SlicerNNUnet/Testing/ModelParityTestCase.py`. Supply the already approved
reference image/prediction and exact model-compatible runtime. They do not
install dependencies, alter research cohorts or select model variants.

For each of `CNN` and `VIT`, set `NNFOUNDATION_<NAME>_MODEL`, `_INPUT`, `_REFERENCE`
and `_DEVICE` before starting Slicer. INPUT is one image path or a JSON array of
ordered channel paths. Optional `_PARAMETERS` points to JSON freezing `folds`,
`checkPointName`, `stepSize` and `disableTta`. Defaults match the D091 examples;
use an explicit file for any other already established recipe. With this
checkout's `SlicerNNUnet` directory on Slicer's Python path, run from its console:

```python
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName("Testing.ModelParityTestCase.ModelParityTestCase")
unittest.TextTestRunner(verbosity=2).run(suite)
```

## Publish only when authorized

The local checkout keeps the official repository as `upstream`; no user fork
remote exists. When remote operations are enabled explicitly:

```bash
gh repo fork KitwareMedical/SlicerNNUnet --remote=false
git remote add origin https://github.com/YOUR_GITHUB_OWNER/SlicerNNUnet.git
git remote -v
git push -u origin feature/nnfoundation-support
```

Do not force-push or load both the fork and original extension in one Slicer
installation. No release/catalogue publication is performed by these commands.
