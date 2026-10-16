---
license: gpl-3.0
library_name: litert
pipeline_tag: image-classification
tags:
  - tflite
  - litert
  - android
  - on-device
  - mobile-sensing
  - faunapulse
  - insects
  - pollinators
  - camera-traps
  - biodiversity
  - biodiversity-monitoring  
  - taxonomy
  - hierarchical-classification
  - efficientnetv2
---

# insectDCT hierarchical classifier V7 (EfficientNetV2-S), LiteRT (fp16) for FaunaPulse

An unofficial conversion of the **insectDCT hierarchical classifier V7** (its EfficientNetV2-S
network) to a LiteRT `.tflite` file plus its class list, made for the
[FaunaPulse](https://github.com/valentinitnelav/fauna-pulse) Android app. insectDCT is the work
of Bjerge et al. (2026), see [Original work](#original-work) below. These files are not
provided or endorsed by the insectDCT authors.

> **This model is for demo purposes for the FaunaPulse app.**
> I did not train this classifier, and its authors did not make it for FaunaPulse.
> They trained it on crops from their own camera traps, mostly of Northern and
> Central European flower visitors. How accurate it is on FaunaPulse crops (other cameras, other
> places, insects in flight, small crops from phone video) is not known. Treat its answers as
> suggestions to check. For identification, FaunaPulse is meant to use
> [BioCLIP 2](https://huggingface.co/Valen-Peregrine/bioclip-2-tflite) with custom label
> packs for now. This `tflite` converted insectDCT classifier is offered as an
> alternative to compare with. Its 104 classes are fixed by its authors' training.

## Files

{{FILES_TABLE}}

- `.tflite`: the network.
- `.fpack`: its class list, which the app needs next to the model (same name). One row per
  level-3 class with the class's place in each of the three outputs and a taxonomy from kingdom
  to species; `Vegetation` is the "none of these" row.
- `.json`: the export manifest (classes of each level, source file and its SHA-256, the check
  below).
- `LICENSE`: insectDCT's licence (GPL-3.0), copied from the authors' repository.

## Original work

- **Authors**: Kim Bjerge, Simon F. A. Wogram, Pau Enric Serra-Marin, Otar Sakhiashvili,
  Andreas Bonnerup Jepsen and Toke T. Høye.
- **Paper**: *InsectDCT: A generalized pipeline for detection, taxonomic classification, and
  tracking of insects in camera-trap recordings*, bioRxiv preprint (2026),
  https://doi.org/10.64898/2026.07.07.736939. The hierarchical classifier's method: Bjerge et al.
  (2023), Ecological Informatics, https://doi.org/10.1016/j.ecoinf.2023.102278.
- **Code and trained models**: https://github.com/kimbjerge/insectDCT (GPL-3.0).
- **Source weights**: `HierarchicalClassifier_EFF2S_V7.pth` (SHA-256
  `520deb4e8f3a58f717ff86b9dadaee67a72e84186b306d43ce5cd814d3178afc`), one of the three networks
  (ConvNeXt-Base, EfficientNetV2-S, ResNet50) in the authors' zip
  `HierarchicalClassifierV7_31032026.zip` (SHA-256
  `da89fdfc64014c1d8889a19bd6baa80eac409600f290ce671327cf1fa19425ad`), which the
  [insectDCT README](https://github.com/kimbjerge/insectDCT/blob/e459ae87ebe39732963e77e0fcdd0a9a28672a08/README.md)
  links on Google Drive (https://drive.google.com/file/d/15oGWBgp3S08k8VK0r2qzMC65uUBFvsPh/view).
  The network file converted here is byte-identical to the one in that zip (same SHA-256,
  checked against a download from that link on 16 October 2026).
- **Version**: V7 is the authors' newer classifier; their default, and the one evaluated in the
  paper, is V6 (80 classes at level 3, 128 px crops). They trained V7 on the V6 pictures plus more
  butterflies and moths, pictures from a Raspberry Pi HQ camera and short video clips, with
  224 px crops. Of the three V7 networks, EfficientNetV2-S is converted here: it runs on the
  tested phone's GPU (ResNet50 does too, ConvNeXt-Base does not), and in the authors' V6 tests
  it is more accurate than ResNet50. The authors describe it as faster and somewhat less
  accurate than ConvNeXt-Base.

## What it is

- **Network**: one network with three outputs ("heads"), one per level: level 1 with 19 groups
  (e.g. `Hymenoptera_bees`, `Diptera`), level 2 with 41 (mostly families), level 3 with 104
  (species, genera and coarser leftovers). Some classes are not insects: Aranaea (spiders),
  Birds, Herpetofauna (reptiles and amphibians), Isopoda (woodlice), Milipedes, Slugs, Snails and
  Vegetation.
- **Input**: one RGB crop `[1, 3, 224, 224]` (NCHW), float32, values 0..1, with no further
  normalisation (as insectDCT's own `ToTensor()`).
- **Output**: `[1, 164]`, the raw scores of the three heads one after the other (19 + 41 + 104);
  not normalised.
- **Conversion**: fp16 weights. Nothing was retrained.
- **Taxonomy in the class list**: the class names and the three levels are insectDCT's own. The
  kingdom-to-species columns were added for FaunaPulse: looked up in the TreeOfLife-200M names
  list (`embeddings/txt_emb_bioclip-2.5-vith14.json`, Gu et al. 2026, CC0 1.0), with a few hand
  fixes, and checked by eye (`classifier_export/taxa/insectdct-cls-v7.csv`). They are not by the
  insectDCT authors.
- **How FaunaPulse uses the scores**: FaunaPulse combines the three heads with its own rule,
  not with insectDCT's per-class thresholds, so its answers can differ from those of the
  insectDCT pipeline. The rule may change between app versions; the FaunaPulse documentation
  describes the current one.

## Checks

- **Agreement with insectDCT's own code** (same model, not an accuracy test): on 34 crops from a
  video clip of bumblebees, insectDCT's own decision code gave the same answer with the phone file
  as with the original network on 34 of 34 crops; the largest score difference was 0.033.
- **Phone** (Xiaomi 11T Pro, Snapdragon 888): 0.02 s per crop on the GPU, 0.13 s on the CPU
  (2 threads); GPU and CPU agree (cosine 0.9998).
- The authors publish per-network test results only for V6 (`metrics/*V6_ClassScoresTest.csv`;
  128 px crops, 80 classes): there, EfficientNetV2-S is ahead of ResNet50 at every level and
  ConvNeXt-Base ahead of both. No such results are published for V7. ConvNeXt-Base does not run
  correctly on the tested phone's GPU.

## Use in FaunaPulse

In FaunaPulse versions after v0.8.0-alpha.1, home screen menu, **Download & import models**
downloads the model and its class list from links pinned to one commit of this repository and
checks each file's SHA-256. With any version you can also download both here and import them by
hand (Import model files…). Choosing the model on the Identify screen chooses its class list.

## How it was made

`classifier_export/export_insectdct_cls.py` of
[fauna-pulse-models](https://github.com/valentinitnelav/fauna-pulse-models/tree/{{CODE_COMMIT}}/classifier_export)
(CPU only, 1 October 2026). It builds the network with insectDCT's own code (the `common/`
folder of the insectDCT repository at commit e459ae8, GPL-3.0, used where it is, not copied),
loads the V7 weights from the authors' zip (unpacked) and converts them:

```bash
python export_insectdct_cls.py --source-dir /path/to/HierarchicalClassifierV7 \
    --code-dir /path/to/insectDCT --backbone eff2s --check-images /path/to/check_crops
```

The network file was written on 1 October 2026 as `insectdct-cls-v7_eff2s_fp16.tflite` (the
script's naming at the time) and is published here under the FaunaPulse naming rule (same
bytes). The class list and the manifest were written again by the script on 16 October 2026
under the new name, from the authors' zip and code downloaded from their links (same check
result), because the class list names the model file it belongs to and the app checks that
name. The `.json` keeps the network's date (`exported`) and records the class list's
(`class_list_written`).

## Limitations

- Trained by its authors on camera-trap crops of flower visitors, mostly Northern and Central
  European taxa. It can only answer with its 104 classes, so a taxon it was not trained on gets
  the most similar class.
- How accurate it is on other cameras, places and kinds of crops is not known.
- The agreement check shows that the conversion matches the original, not how well it names
  insects in your setting.

## Licence

GPL-3.0, the licence of the insectDCT repository. The authors share the V7 classifier weights
through a Google Drive link in that repository's README without a separate licence. This
conversion treats them as GPL-3.0, the licence of the repository that links them. A copy of the licence is in
`LICENSE`. Changes made here: converted to LiteRT with fp16 weights on 1 October 2026 (nothing was
retrained), and the taxonomy columns of the class list added. The conversion code is in
fauna-pulse-models repository (AGPL-3.0), linked above.

## Citation

If you use these files, please cite the insectDCT paper:

Bjerge, K., Wogram, S. F. A., Serra-Marin, P. E., Sakhiashvili, O., Jepsen, A. B., & Høye, T. T.
(2026). InsectDCT: A generalized pipeline for detection, taxonomic classification, and tracking
of insects in camera-trap recordings. *bioRxiv* (preprint, not peer reviewed).
https://doi.org/10.64898/2026.07.07.736939

```bibtex
@article{bjerge2026insectdct,
  author       = {Bjerge, Kim and Wogram, Simon F. A. and Serra-Marin, Pau Enric and Sakhiashvili, Otar and Jepsen, Andreas Bonnerup and H{\o}ye, Toke T.},
  title        = {{InsectDCT}: A generalized pipeline for detection, taxonomic classification, and tracking of insects in camera-trap recordings},
  journal      = {bioRxiv},
  elocation-id = {2026.07.07.736939},
  year         = {2026},
  doi          = {10.64898/2026.07.07.736939},
  note         = {Preprint}
}
```

Also relevant:

- Method of the hierarchical classifier: Bjerge, K., Geissmann, Q., Alison, J., Mann, H. M. R.,
  Høye, T. T., Dyrmann, M., & Karstoft, H. (2023). Hierarchical classification of insects with
  multitask learning and anomaly detection. *Ecological Informatics*, 77, 102278.
  https://doi.org/10.1016/j.ecoinf.2023.102278
- Taxonomy columns of the class list: Gu, J., Stevens, S., Campolongo, E. G., et al. (2026).
  TreeOfLife-200M (Revision 94bbc0b) [Data set]. Hugging Face. https://doi.org/10.57967/hf/8980

## Notes

- Files in this repository are never replaced: the app pins each download to a commit and
  checks its SHA-256. New weights get a new file name.
