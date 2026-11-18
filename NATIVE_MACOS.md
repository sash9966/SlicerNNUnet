# Local macOS native runtime

The first validation target is Slicer 5.12.3 on macOS 27.0.1 arm64, Apple M5 Max
with 128 GiB unified memory, using CPU explicitly. The original CNN and ViT checkpoints both
restore through the current official native nnU-Net loader, with exact equality
of every loaded state tensor and unchanged checkpoint hashes. No checkpoint
retagging, new trainer classes, plans changes or nnssl installation is needed.
See [VALIDATION.md](VALIDATION.md) for prediction and workflow results.

## Runtime sources

The installed nnunetv2 2.8.1 distribution lacks the required original pretraining
modules. The tested native implementation is official nnU-Net source commit
`131289433bdac0610f8b80a0b406e18bcfe51a2a` (source version 2.9.0), plus official
batchgenerators source commit `acc03595d5131f160bce20629e707f3178afd8c4`.
The latter supplies `SEG_TIEBREAKS`, absent from installed batchgenerators 0.25.3.

These sources are selected for the inference subprocess. Slicer's installed
packages remain unchanged: Torch 2.2.2, dynamic-network-architectures 0.4.4,
timm 1.0.22 and NumPy 1.26.4. Installed distribution metadata still reports
nnunetv2 2.8.1; record the imported source commit as well as package metadata.
This is an inference-specific tested combination, not a training environment.

Keep the checkouts outside the installed extension:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git /path/to/runtime/nnUNet
git -C /path/to/runtime/nnUNet checkout --detach 131289433bdac0610f8b80a0b406e18bcfe51a2a
git clone https://github.com/MIC-DKFZ/batchgenerators.git /path/to/runtime/batchgenerators
git -C /path/to/runtime/batchgenerators checkout --detach acc03595d5131f160bce20629e707f3178afd8c4
```

On 17 November 2026, nnunetv2 2.9.0 was not published on PyPI. Entering
`==2.9.0` in the existing installer therefore cannot obtain this source revision.
The installer also does not preserve direct Git URL requirements. This validation
uses source selection instead of changing that installation mechanism.
[Published nnunetv2 releases](https://pypi.org/project/nnunetv2/),
[published batchgenerators 0.25.6](https://pypi.org/project/batchgenerators/0.25.6/).

## Select the runtime in Slicer

After Slicer starts, use its Python console to configure child inference processes:

```python
import os

os.environ["PYTHONPATH"] = os.pathsep.join([
    "/path/to/runtime/nnUNet",
    "/path/to/runtime/batchgenerators",
    os.environ.get("PYTHONPATH", ""),
])
os.environ["nnUNet_extTrainer"] = os.pathsep.join([
    "/path/to/CNN/ExistingTrainer__ExistingPlans__3d_fullres",
    "/path/to/ViT/ExistingTrainer__ExistingPlans__3d_fullres",
])
os.environ["nnUNet_compile"] = "false"
```

Use the ordinary run directories containing each export's unchanged original
`paper150_trainers.py` and `paper_trainers.py`. Official `nnUNet_extTrainer`
discovery imports those existing classes; the checkpoint selects the actual
trainer. Both supplied exports contain byte-identical copies of these sources.

The explicit standard `nnUNetv2_predict` script in the tested installation uses
Slicer's `python-real` interpreter and inherits this source-first environment.
No replacement executable or launcher wrapper is used. Setting `PYTHONPATH`
only before launching PythonSlicer was insufficient because its launcher sets
its own search path; set the child environment after Slicer starts. Standalone
PythonSlicer diagnostics insert these source directories into `sys.path` before
importing nnU-Net.

These environment assignments apply to the current Slicer session. Registered
CNN/ViT paths and inference preferences persist through QSettings independently.
The default installed runtime and startup configuration have not been changed.
On the validation Mac, the verified local paths/settings have been saved under
`SlicerNNUNet/RegisteredModels/CNN` and `/ViT`. Existing manual parameters and the
current selection were preserved. A fresh installation still needs its own local
model registration.

Select CPU, native fold `all`, `checkpoint_final.pth`, step 0.5 and disabled TTA
for the recorded local check. Normalization, spacing, patch dimensions and label
identities come from the unchanged export. MPS performance and compatibility are
separate checks.

The local prepared source checkouts are beside this repository in
`../SlicerNNUnet-runtime/nnUNet` and `../SlicerNNUnet-runtime/batchgenerators`.
Weights and private input/output records are outside this Git repository.
