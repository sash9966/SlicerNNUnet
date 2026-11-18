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

Checks performed on 17 November 2026. The user selected local macOS as the first
validation target: Slicer 5.12.3, Python 3.12.10, macOS 27.0.1 arm64, Apple M5 Max,
128 GiB unified memory, CPU selected explicitly. Other colleague platforms have
not been tested.

Final result: **57 fast tests passed** (28 Parameter, 24 process/import safety,
5 registry); **23 actual Slicer tests passed** (5 installer metadata, 4 real
QProcess, 4 synthetic geometry/scene safety, 10 widget). The two opt-in strict
parity unit tests initially reported missing prerequisites. Subsequent private
actual-model workflow checks used one shared local scan and the two matched
saved predictions; their results are listed separately below.

| Check | Status | Evidence / limit |
|---|---|---|
| Standard configuration, fold `0`, explicit ensemble, native `all` | PASS | 28 standard-Python configuration tests |
| Missing folds/checkpoints, invalid/mixed folds, `__` identifiers, spaced paths, malformed metadata | PASS | Same configuration suite |
| Independent CNN/ViT/Manual settings, restart reconstruction, explicit path update | PASS | Standard-Python registry suite and real Slicer widget tests |
| Actual CNN/ViT registration after restarting Slicer | PASS | Fresh isolated Slicer process restored both paths, native all/final checkpoint, CPU, step 0.5 and disabled TTA from the same settings file; selecting entries did not run inference |
| Verified paths registered in local Slicer preferences | PASS | CNN/ViT entries saved and read back in the real local QSettings file; existing manual parameters and current selection preserved |
| Mixed original/fork installation detection | PASS | Actual installed original library is rejected with an actionable ImportError; matching fork library imports normally |
| Process errors, crash, cancellation, stale output and reentrant startup | PASS | Fake-process regression suite; actual widget cancellation checks |
| Real Qt subprocess success, memory/dependency error and cancellation | PASS | Four actual QProcess tests in Slicer, including its one-argument finished-signal binding |
| Apply does not install or change dependencies | PASS | Actual Slicer widget tests; InstallLogic.py unchanged |
| Oblique/anisotropic geometry and parent transform | PASS | Actual Slicer NIfTI export/read, segmentation import and labelmap world-affine check; affine tolerance 1e-5 |
| Segment names and sparse IDs | PASS | Actual Slicer IDs 1/7, metadata names and exact per-segment voxel masks |
| Source scan and existing segmentation preserved on failure/Stop | PASS | Actual Slicer synthetic scene tests |
| Both supplied exports' metadata and native CLI arguments | PASS | Actual Slicer Parameter.isValid/asArgList, native `all`, exact original identifiers/final checkpoint, TTA disabled, step 0.5; no tensor load |
| Original trainer in default installed nnunetv2 2.8.1 | FAIL | Required pretraining modules are absent; installed packages were left unchanged |
| Both original checkpoints through current official native source | PASS | Strict restoration, exact state-tensor equality: CNN 956 tensors; ViT 936 tensors; original checkpoint SHA256 unchanged |
| Actual CNN/ViT Apply, editable import, edit/revert, save/reload | PASS | One shared scan, both models; all seven metadata label IDs/names preserved; output/reference/input affines equal exactly; all inspected source hashes unchanged |
| New native CNN versus original inference implementation on the same CPU | PASS | Exactly identical labels and geometry; no accepted numerical tolerance |
| New native ViT versus original inference implementation on the same CPU | PASS | Exactly identical labels and geometry; no accepted numerical tolerance |
| Exact equality to historical CUDA predictions | FAIL | CNN 272 and ViT 349 differing voxels out of 29,360,128 per output; tolerance has not been relaxed |
| Standard nnU-Net actual weighted-reference regression | UNRUN | A stock model is local, but its matched reference prediction was not retained locally; configuration and synthetic I/O checks passed |
| MPS/GPU inference and real GPU-memory failures | UNRUN | CPU is the validated local device; failure diagnostics also tested with fakes |
| Extension package build/install and other colleague environments | UNRUN | Actual workflow used source-loaded extension code in an isolated Slicer process; no extension package built or shared runtime changed |

Synthetic/fake tests establish configuration and I/O behavior. Actual-model checks
used one shared scan for technical validation; clinical accuracy was not assessed.
The opt-in parity tests retain identical-label requirements and affine tolerance
1e-5; actual input, output and reference affine differences were zero.

Both controls ran the unchanged established `predict_pretrained.py` with exact
reference nnUNet source `202f6baa0adc2ef5f7b615df19cc4da970412cc0`, using the same
local CPU, Torch 2.2.2 and DNA 0.4.4. They reproduced the new native CPU outputs
exactly and showed the same 272/349-voxel differences from the saved CUDA outputs.
The new loading route therefore does not cause those differences. Historical
CUDA inference used Torch 2.6.0 with CUDA autocast; CPU inference uses float32
network computation. This control does not establish historical CUDA bit parity,
and no nonzero label tolerance has been accepted.

Both models passed actual Segment Editor assignment, reversible mask editing,
`.seg.nrrd` save/reload, original image preservation and metadata label checks.
The registered settings were also read successfully in a fresh Slicer process.
This source-loaded widget workflow does not certify package installation: a
separate source-module loading check with the original extension still installed
selected its library instead of the fork. A small origin guard now rejects that
mixed installation clearly; actual matching-source import passed. Disable/uninstall
the original before loading this fork, as described in the README. User installation
settings were left unchanged.

Private scripts, source bindings, checkpoint hashes, runtime logs, output masks,
editable results and fresh-process receipts are retained outside Git in the sibling
`SlicerNNUnet-local-validation/20261117-macos` directory. The two public runtime
checkouts and the historical reference checkout are in `SlicerNNUnet-runtime`.

## Native runtime requirement

Both D091 exports retain original `PretrainedTrainer_CT_Adam150` /
`PretrainedTrainer_CT_ViT_Adam150` trainers and `ResEncL` / `PrimusX` architecture
identifiers. The original trainer modules already accompanying the exports are
discovered through official `nnUNet_extTrainer`, without new trainer classes or
distribution wrappers.

Their historical reference nnUNet commit
`202f6baa0adc2ef5f7b615df19cc4da970412cc0` also has a native builder-signature
gap: the CNN builder requires `input_patch_size`, while that native predictor
does not supply it; Primus receives no saved patch size. The already established
original inference implementation supplies the saved patch size explicitly.
Installing the historical package alone therefore does not certify native
prediction compatibility.

Current official nnU-Net fixes the native builder signatures. Both original
checkpoints now pass native strict restoration, and both complete the actual
editable-result workflow using that implementation. The tested source-only
runtime is pinned nnUNet `131289433bdac0610f8b80a0b406e18bcfe51a2a` plus
batchgenerators `acc03595d5131f160bce20629e707f3178afd8c4`, selected only for
inference subprocesses in existing Slicer Python. Installed package metadata
remains nnunetv2 2.8.1; the executing source is version 2.9.0. No nnssl package,
shared dependency upgrade or source-export edit was required.

The 2.9.0 source is not published on PyPI as of this check, so the standard
installer cannot obtain it with `==2.9.0`. The reproducible local source recipe
and its per-session setup are in [NATIVE_MACOS.md](NATIVE_MACOS.md).
**Recommendation: use that pinned CPU runtime for local review; preserve exact
reference-parity failures and the unrun deployment/GPU checks.**

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
