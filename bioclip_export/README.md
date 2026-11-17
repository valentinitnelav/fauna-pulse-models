# BioCLIP export tools: put BioCLIP 2 on the phone (PC side)

FaunaPulse's *Identify organisms* feature runs the BioCLIP 2 image tower on the phone.
The phone cannot read the original PyTorch checkpoint, so two files are prepared once
on a normal computer (no GPU needed) and copied to the phone:

| File | What it is | Size (BioCLIP 2) |
|---|---|---|
| `bioclip-2_224_fp16.tflite` | the image tower converted for the phone (fp16 weights) | 581 MiB |
| `<pack>.fpack` | a **label pack**: the names the model may choose from, their pre-computed embeddings, their taxonomy, plus six "none of these" entries (flower, leaf, shadow, debris, blurry, web) | 60 MiB for 32 flower-visitor families; up to ~430 MB for all Insecta + Arachnida |

Everything below was run end to end on 2026-09-20 (Ubuntu 24.04, Python 3.12, 15 GB RAM,
no GPU). Times and sizes are from that run.

## 0. What you need

- Python 3.11 or 3.12 (tested: 3.12.3). 3.13 is not yet supported by the converter.
- RAM: about 7 GB free during the conversion of BioCLIP 2 (BioCLIP 2.5 needs roughly twice).
- Disk: about 12 GB free in total: 4 GB for the Python packages (CPU build of PyTorch),
  1.7 GB for the BioCLIP 2 checkpoint, 2.8 GB for the TreeOfLife name embeddings (both
  downloaded once into `~/.cache/huggingface`), and 2 GB for the conversion outputs.
- Internet for the first run (Hugging Face downloads; no account or token needed).
- `adb` (Android platform tools) to copy files to the phone, or any USB file transfer.

## 1. Create the Python environment

All three tools share one environment, made once from the repository root as described in
the [repository README](../README.md#setup). Then, from the repository root:

```bash
source .venvs/export/bin/activate
cd bioclip_export
```

The environment uses the CPU build of PyTorch (about 300 MB instead of the 3.5 GB CUDA
build); both give identical results here.

## 2. Export the image tower (about 2 minutes plus the download)

```bash
python export_image_tower.py --model bioclip-2 --precision fp16 --out ./out
```

What happens: the 1.7 GB checkpoint `imageomics/bioclip-2` is downloaded into the
Hugging Face cache (once); the image tower is wrapped so the phone can feed plain RGB
pixels in 0..1 (the colour normalisation and the final L2 normalisation are inside the
graph); `litert-torch` converts it to a float32 `.tflite` (1,216 MB, 74 s, about 7 GB
RAM); `ai-edge-quantizer` casts the weights to fp16 (609 MB, 6 s). Output:

```
out/bioclip-2_224_fp16.tflite   the model for the phone
out/bioclip-2_224_fp16.json     manifest: embedding size 768, input 224x224, logit scale 100, sha256
```

**GPU-friendly attention (round 242, the default).** Every attention layer is exported as
plain matrix steps with tensors of at most 4 dimensions (`--attention 4d`). PyTorch's own
attention layer passes the data through 5-dimensional tensors, which the phone's GPU engine
cannot run: files exported before round 242 therefore always fall back to the CPU (the
phone's log says `RESHAPE ... has bad input dims size`; only 63 of 1488 steps could go to
the GPU). With the 4d export all 1392 steps run on the GPU. The weights and the maths are
the same. On the Xiaomi test phone one crop took 0.27 s on the GPU instead of
2.6 s on the CPU. `--attention torch` makes the old kind of file (for comparison only).
The manifest records which kind a file is (`"attention": "4d"`).

Options: `--precision fp32` (nothing cast, 1.2 GB; also runs on the phone),
`--precision int8` or `int8-weights` (about 0.3 GB; see section 2b before using them),
`--keep-fp32` (keep the float32 intermediate, useful for `verify_parity.py`),
`--model bioclip-2.5` (ViT-H/14, 3.9 GB download, 1.27 GB fp16, about 11 GB RAM; section 2c),
`--weights FILE` (a checkpoint already on disk instead of the Hugging Face download),
`--onnx` (additionally write an ONNX file).

Check the file before copying it anywhere:

```bash
python inspect_tflite.py out/bioclip-2_224_fp16.tflite
```

Expected (4d export): `FULLY_CONNECTED: {'fp16 weights': 145}` (every large layer is
fp16; the attention's in-projection is split into three), input
`serving_default_args_0 [1, 3, 224, 224]`, output `serving_default_output_0_output [1, 768]`.
If it says `weights computed at runtime (unfolded)`, the conversion ran in the
memory-saving mode (see Troubleshooting).

## 2b. Quantisation: what it is and when it makes a model faster

The float32 file stores each of BioCLIP 2's ~300 million weights in 4 bytes.
*Quantising* stores them with fewer bits: the file gets smaller, and some chips compute
faster, at the price of a little rounding. `quantise_tflite.py` does this step (the
export calls it) and works on any float32 `.tflite`, so collaborators can use it for
their own classifiers or embedders. For YOLO detectors, use Ultralytics' own export
instead ([`docs/MODEL_CONVERSION.md`](../docs/MODEL_CONVERSION.md)). Its comments explain each kind in more detail.

| Kind | Size (BioCLIP 2) | Faster where |
|---|---|---|
| fp32 | 1216 MB | nowhere (the baseline) |
| **fp16** (default) | 609 MB | phone GPU (its native 16-bit maths); on a CPU the weights are widened back to 32 bits, so same speed |
| int8-weights | 309 MB | nowhere: weights widened to floats on loading; only the file is smaller |
| int8 (dynamic range) | 309 MB | CPU (8-bit maths); the gain depends on the chip |

Fewer bits round more: fp16 stays closest to the original, int8 (8-bit maths) changes the
outputs the most; check a quantised file with `verify_parity.py` on your own crops. On a laptop CPU (Intel i5-8350U, 4 threads) all four took 1.0 to 2.0 s per crop:
int8 was at most about 20 % faster, which is less than the laptop's own run-to-run
spread. On the Xiaomi test phone (Snapdragon 888) fp16 took 0.27 s per crop on the GPU
and 2.6 s on the CPU. int8 on a phone CPU has not been measured yet.

**Does quantisation need a GPU?** Making the file does not: it runs on any computer's
CPU in seconds. Whether the smaller file then *runs* faster depends on the chip that
runs it. fp16 helps a GPU. On a CPU only int8 with 8-bit maths can help, and for
BioCLIP it cost accuracy on unsure crops. That is why the app ships fp16 and uses the
GPU where it can.

## 2c. BioCLIP 2.5 (round 264)

BioCLIP 2.5 Huge (ViT-H/14, embeddings of 1024 numbers) needs its own model file and its
own label packs (BioCLIP 2 packs do not fit it). Without `--weights` and `--embeddings-dir` the
scripts download the authors' files from Hugging Face. When the checkpoint
(`open_clip_model.safetensors` of imageomics/bioclip-2.5-vith14) and the TreeOfLife name
embeddings (`embeddings/txt_emb_bioclip-2.5-vith14.npy` and `.json` of the dataset
imageomics/TreeOfLife-200M) are already on disk in one folder, point the scripts at them
instead of downloading 7 GB again:

```bash
W=/path/to/bioclip-2.5
python export_image_tower.py --model bioclip-2.5 --weights $W/open_clip_model.safetensors --out out/bioclip25
python build_label_pack.py --model bioclip-2.5 --embeddings-dir $W --weights $W/open_clip_model.safetensors \
  --classes Insecta --species-csv out/species_europe_pollinator_orders.csv \
  --pack-id bioclip-2.5_pollinator-orders-europe_v1 --out out/bioclip25
python verify_parity.py --tflite out/bioclip25/bioclip-2.5_224_fp16.tflite --weights $W/open_clip_model.safetensors \
  --images /path/to/crops --pack out/bioclip25/bioclip-2.5_pollinator-orders-europe_v1.fpack
```

Measured on the 15 GB laptop (2026-10-01): export 4 minutes, 10.7 GB RAM at the peak (it
used swap); the float32 intermediate is 2.5 GB, above the 2 GB limit of a plain `.tflite`,
which the converter and the quantiser handle by themselves; the fp16 file is 1,266 MB with
all 193 large layers in fp16 and all 32 attention layers GPU-friendly. Packs built:
`bioclip-2.5_pollinator-orders-europe_v1` (34,704 names, 74 MB) and
`bioclip-2.5_flower-visitors-32fam_v1` (37,461 names, 80 MB); the 2.5 TreeOfLife table has
794,878 names (BioCLIP 2's: 867,455), so the same filters keep slightly different lists.
On the phone: the app's [`docs/IDENTIFICATION.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/IDENTIFICATION.md), *BioCLIP 2.5* (CPU only on the 7.4 GB test phone,
5.3 to 5.6 s per crop).

int8 was tried too (`--precision int8`, 639 MB): it changed the outputs more than fp16, and on
the phone's CPU it was slower than fp16 (9.2 s against 5.6 s per crop). fp16 stays the choice.

## 3. Build a label pack (about 5 minutes plus the download)

A pack from a taxon list given by the user, here the first pack for a phone test: every
species, worldwide, of 32 hand-picked families of flower visitors (bees, wasps, hoverflies
and other flies, ladybirds and other beetles, butterflies, bugs, lacewings, scorpionflies,
crab, orb and jumping spiders), 38,570 species, 60 MiB. It is a demo of the option, not a
justified selection: of its insects in the four main orders only 14 % have 3 or more GBIF
records in Europe, and it leaves out flower visitors such as the bee family Melittidae,
longhorn beetles and most moths. It is not maintained (no v2); for Europe use the regional
pack of 3b.

```bash
python build_label_pack.py --model bioclip-2 --pack-id bioclip-2_flower-visitors-32fam_v1 --out ./out \
  --families Apidae,Halictidae,Andrenidae,Megachilidae,Colletidae,Vespidae,Syrphidae,Muscidae,Calliphoridae,Sarcophagidae,Bombyliidae,Empididae,Conopidae,Stratiomyidae,Tachinidae,Coccinellidae,Oedemeridae,Cantharidae,Pieridae,Nymphalidae,Lycaenidae,Hesperiidae,Papilionidae,Zygaenidae,Sphingidae,Pentatomidae,Miridae,Panorpidae,Chrysopidae,Thomisidae,Araneidae,Salticidae
```

What happens: the TreeOfLife-200M name embeddings (`txt_emb_species.npy`, 2.66 GB, plus
a 92 MB json with the taxonomy) are downloaded once into the Hugging Face cache; the
rows of the requested families are kept (867,455 species in total; Insecta 264,036,
Arachnida 16,406); the six "none of these" prompts are embedded with the BioCLIP text
tower; everything is written as one `.fpack` file (format documented in `fpack.py`). Since
round 266 the same container also holds the *class lists* of fixed-class classifiers such as
insectDCT (no name vectors; `fpack.write_class_list`, see [`classifier_export/README.md`](../classifier_export/README.md)).

### 3b. Regional packs (GBIF occurrence lists)

`build_region_species_list.py` writes the species of chosen orders that GBIF has at
least 3 records for in a region (a GBIF continent or a set of countries). Restricting
the candidates to a regional GBIF species list is what BioCLIP's documentation
recommends ("Geo-Restricted Taxon List Predictions"). The way of getting that list follows
Maximilian Sittinger's `insect-detect-post`:
records are counted per accepted species of the Catalogue of Life Extended Release (the
taxonomy GBIF uses since it replaced the GBIF Backbone), and TreeOfLife names are mapped to
those species with his mapping table (release v1.1.0, made from the BioCLIP 2 names file and
COL26.6 XR; downloaded on first use and checked by SHA-256). The CSV also gives the
accepted COL name of each species for reference; the pack keeps the TreeOfLife names. The list
holds BioCLIP 2 strings, so a BioCLIP 2.5 pack keeps only the names its own file writes the same
way (Europe, 26 October 2026: 586 of 34,974 names have no BioCLIP 2.5 row, some only spelled
differently). One GBIF request per region, a few seconds each, no account needed:

```bash
python build_region_species_list.py --continent EUROPE   --orders Diptera,Hymenoptera,Coleoptera,Lepidoptera --out out/species_europe_4orders_col.csv
python build_label_pack.py --model bioclip-2 --classes Insecta   --species-csv out/species_europe_4orders_col.csv --pack-id bioclip-2_pollinator-orders-europe_v2 --out ./out
```

Countries instead of a continent: `--countries DE,AT,CH,CZ,PL` (union; each country on its
own needs the 3 records). Any CSV with a `species` column of "Genus epithet" names works as
`--species-csv`, so a user-supplied taxon list is the same one-liner. The v1 packs below used
the earlier mapping (v1.0.0, GBIF Backbone keys, one request per order); the script no longer
rebuilds their species lists (written to `out/` when those packs were built; file names in
each pack header, `source.species_csv`).

**GBIF keys and country lists (v3 packs, since 2026-10-27).** A species CSV of
`build_region_species_list.py` also gives every pack row its GBIF species key and accepted
name (header block `gbif`, one entry per row; the same mapping as insect-detect-post, so a
track gets the same key in both programs and joins GBIF data, synonyms included). With
`--country-lists` the pack also records which of its rows GBIF lists in each country (at least 3
presence records there, no fossil specimens) and each country's neighbours (block `regions`), so
the app can narrow the names to the session's country and its neighbours, or use the whole
pack. Format of both blocks: `fpack.py`; older app versions ignore them.

```bash
python build_country_lists.py --species-csv out/packs_v2/species_europe_4orders_col.csv --out out/packs_v3/countries_4orders_col.json
python add_pack_blocks.py out/packs_v2/bioclip-2_pollinator-orders-europe_v2.fpack --species-csv out/packs_v2/species_europe_4orders_col.csv \
  --country-lists out/packs_v3/countries_4orders_col.json --pack-id bioclip-2_pollinator-orders-europe_v3 --out out/packs_v3
python check_pack_blocks.py out/packs_v3/bioclip-2_pollinator-orders-europe_v3.fpack --same-as out/packs_v2/bioclip-2_pollinator-orders-europe_v2.fpack
```

`add_pack_blocks.py` copies an existing pack's rows and vectors byte for byte and adds the
blocks (seconds, no torch). A new pack gets them from `build_label_pack.py --country-lists FILE`
(same blocks); it is not used for v3 because it embeds the six "none of these" rows again, and
their float16 values can differ in the last bit from one computer to another (3 of 6 rows, at
most 3e-5, against v2). `build_country_lists.py` sends one GBIF request per country (250
countries, about 3 minutes).

Only presence records count and fossil specimens are left out (since 2026-10-28,
`record_filter()` in `build_region_species_list.py`, also for new Europe lists). The v2 Europe
list is older: 163 of its names would not be in it today; v3 keeps them, as it keeps all of v2's rows.

Neighbours come from the tracked table `country_neighbours.csv` (425 pairs): a shared land
border, or at most 100 km apart over the sea by a connection that crosses no third country
(Germany and Sweden, Great Britain and France, Malta and Italy, Croatia and Italy over the Gulf
of Trieste; not Germany and Italy, 46 km apart across Austria; Iceland has none). Two pairs are
added by hand, with their reason in `country_neighbours.json`: Greece and Italy (Natural Earth
lacks the small Greek islands about 80 km from Italy) and Anguilla and Sint Maarten (French
Saint-Martin lies between them on the same island). Computed from
Natural Earth (map units 1:10m, v5.1.2, public domain) by `country_neighbours.py` (needs
shapely; rerun only to change the rule). Natural Earth units without an ISO code (e.g. Northern
Cyprus) take the code GBIF's reverse geocoder gives them. How much a country narrows the Europe
list (v3): see 3c.

**The whole world, and lists per continent (November 2026).** `--world` counts the records of
the whole world instead of a region (GBIF does not answer one request for all arthropods of the
world, so the count is made order by order). `--min-occurrences 1` keeps every species GBIF has a
presence record of. `--leave-out-extinct` leaves out species that the Catalogue of Life marks as
extinct and that GBIF holds no observation of a living individual of; the mark alone also sits
on some living species, and the fossil-specimen rule alone misses fossils filed as preserved
specimens. The names left out are written to `<list>_extinct_left_out.csv` (961 for the world
list, nearly all amber ants and fossil flies and beetles). One country-list file serves every
list of the same orders:

```bash
M=out/mapping_col26.10xr/bioclip_gbif_species_mapping_Arthropoda.csv   # see the next paragraph
O=Diptera,Hymenoptera,Coleoptera,Lepidoptera
P=out/packs_2026-11
python build_region_species_list.py --world --min-occurrences 1 --leave-out-extinct --orders $O --mapping-csv $M --out $P/species_world_4orders.csv
python build_region_species_list.py --continent north_america --leave-out-extinct --orders $O --mapping-csv $M --out $P/species_north-america_4orders.csv
python build_country_lists.py --species-csv $P/species_north-america_4orders.csv --min-occurrences 3 --mapping-csv $M --out $P/countries_4orders.json
python build_label_pack.py --model bioclip-2 --classes Insecta --species-csv $P/species_north-america_4orders.csv \
  --country-lists $P/countries_4orders.json --pack-id bioclip-2_pollinator-orders-north-america_v1 --out $P
python check_pack_blocks.py $P/bioclip-2_pollinator-orders-north-america_v1.fpack --countries US,CA,MX
```

The continents are GBIF's (`europe`, `north_america` with Central America and the Caribbean,
`south_america`, `africa`, `asia`, `oceania`; boundaries: https://github.com/gbif/continents). A
country near a continent border can have many of its species in the neighbouring continent's
list only (Indonesia: 4,665 of its 5,523 names are in the Asia list, 2,191 in Oceania's); the
app warns when a list holds under 80 % of a country's GBIF list.

**When GBIF moves to a newer Catalogue of Life release.** Species keys belong to one release:
when GBIF moved from COL26.6 XR to COL26.10 XR in November 2026, 929 of the 34,974 names of
the Europe list got another key although their names stayed (*Speyeria aglaja*: `93N9L`, then
`4YKG8`), so a mapping of the older release misses such species in GBIF's counts.
`build_region_species_list.py` and `build_country_lists.py` therefore stop when GBIF's release
is not the mapping's. `make_gbif_mapping.py` then makes the mapping for the current release by
running insect-detect-post's own mapping script unchanged (it needs his environment; about
300,000 name requests to GBIF, 7.5 hours with the retry passes on 17 November 2026;
`--model bioclip-2.5` maps that model's names file), and `--mapping-csv` gives it to the two
scripts. Lists and packs record the mapping they were made with (SHA-256, COL release, the
commit of insect-detect-post). A published pack is not affected: its names and country lists
stay as built, and its keys are those of the release named in its header.

**Mammals and birds (November 2026).** Insect-detect-post's mapping covers only arthropods, so
`make_gbif_mapping.py --phylum Chordata` runs his script unchanged on the chordates of the
BioCLIP 2 names file (57,900 names, 53,774 with a key; 20 minutes with 4 workers). `--classes`
replaces `--orders` in both list scripts (GBIF is then asked for the records of those classes), and
`--every-mapped-name` writes every mapped name without a region: the key list of a pack that keeps
all its rows, as the worldwide mammals-and-birds pack does (`add_pack_blocks.py` takes it for a pack
built from whole classes):

```bash
M=out/mapping_col26.10xr_chordata/bioclip_gbif_species_mapping_Chordata.csv
P=out/packs_2026-11
python build_region_species_list.py --every-mapped-name --classes Mammalia,Aves --mapping-csv $M --out $P/species_mammals-birds_keys.csv
python build_country_lists.py --species-csv $P/species_mammals-birds_keys.csv --min-occurrences 3 --mapping-csv $M \
  --leave-out-extinct --leave-out-iucn-extinct --leave-out-pbdb-extinct --mdd-csv MDD_v2.5_6904species.csv \
  --out $P/countries_mammals-birds.json
python add_pack_blocks.py bioclip-2_mammals-birds-world_v2.fpack --species-csv $P/species_mammals-birds_keys.csv \
  --country-lists $P/countries_mammals-birds.json --in-every-country people-and-domestic --short-list-names 300 \
  --drop-sink person --pack-id bioclip-2_mammals-birds-world_v3 --out $P
```

Extinct species stay rows of the pack, and three rules keep them off the country lists (the names
and the rule that took each go to `<lists>_extinct_left_out.csv`): the two-part rule above
(`--leave-out-extinct`); the IUCN Red List category EX (`--leave-out-iucn-extinct`, from the
mapping; "extinct in the wild" stays, such species are reintroduced); and the Paleobiology Database
(`--leave-out-pbdb-extinct`, https://paleobiodb.org, CC BY 4.0), for Ice Age species such as
*Mammuthus primigenius* and *Ursus spelaeus*, which GBIF holds as museum specimens and material
samples (not "fossil specimens") and which neither the Catalogue of Life nor the IUCN marks. PBDB
alone also answered "extinct" for living species whose name matches a fossil (the moose, the anoas),
so a species goes only when, in addition, the IUCN has not assessed it as living, under 10 % of its
GBIF records are observations, and, for mammals, the Mammal Diversity Database v2.5
(https://doi.org/10.5281/zenodo.21654811, CC BY 4.0; the file is checked by SHA-256) does not list it
as a living species. People and domestic animals live everywhere but are rarely recorded (at least
3 records: *Homo sapiens* in 67 countries of 250, the TreeOfLife row *Canis lupus*, common name
Domestic Dog, in 78), so `--in-every-country people-and-domestic` puts ten rows on every country's
list (humans, dog, cat, cattle, sheep, goat, pig, horse, donkey, chicken; `EVERY_COUNTRY_SETS` in
`build_label_pack.py`, recorded in the header). Since v3 the `mammal` sink set has no "person" row:
the app sends MegaDetector's "person" boxes to BioCLIP, and the *Homo sapiens* row names them.
`--short-list-names` records below how many names a country with its neighbours counts as short
for this pack (`regions.short_list_names`); the app's own rule (5,000 names) fits the pollinator
lists, while a mammals-and-birds country with its neighbours has 25 to 4,751 names.

How complete the mammal lists are, against the Mammal Diversity Database's country field (v2.5,
the 3,912 MDD species whose name matches a mapped TreeOfLife mammal): a country with its neighbours
holds a median 97 % of its MDD mammals (the lowest tenth of countries under 87 %), a country alone
78 %. Lists with neighbours under 300 names are nearly all small islands, where most of the MDD
species missing are whales and dolphins. Birds are recorded far better (eBird). The lists also hold
more than the species living there (Germany: 1,622 names, 102 MDD mammals), from museum
specimens, vagrants and kept animals.

**Limits.** A country list says where GBIF holds records of a species, and records follow where
people observe, collect and digitise. Europe, North America and Australia are well recorded; much
of the tropics, Africa and Asia is not (e.g.: names of the four orders with at least 3 records, 28
October 2026: Germany 16,768, Brazil 10,204, DR Congo 1,931, Nigeria 1,307), so there a list
misses many species that live in the country, and the neighbours or the whole pack are the
better choice. Species do not follow borders either: a country list holds species of all its
habitats, and a site near a border may share species with a country that is not a neighbour
by this rule. The neighbour rule (100 km, map units 1:10m, distances on a local flat projection)
is one simple choice among many, not an ecological boundary.

### 3c. Packs built so far

November 2026 (built 2026-11-17; the six continent packs published on Hugging Face on
2026-11-17, the world pack not, since no app version opens it yet): the four orders for the whole world and
per continent, made with the mapping of COL26.10 XR and GBIF as queried that day (presence
records, no fossil specimens, extinct species without an observation left out,
`--name-rules higher-rank`), each with the `gbif` and `regions` blocks:

| Pack id | Region | Names | Size (header) | Rows in no country list | Opens in the app |
|---|---|---|---|---|---|
| `bioclip-2_pollinator-orders-world_v2` | world, at least 1 record | 192,886 | 327.7 MB (31.4 MB) | 36,250 | no (header above the app's 16 MB limit): for PC use |
| `bioclip-2_pollinator-orders-europe_v4` | Europe, at least 3 records | 35,014 | 59.6 MB (5.8 MB) | 457 | yes |
| `bioclip-2_pollinator-orders-north-america_v1` | North America, at least 3 | 56,446 | 96.0 MB (9.2 MB) | 425 | yes |
| `bioclip-2_pollinator-orders-south-america_v1` | South America, at least 3 | 23,660 | 40.2 MB (3.9 MB) | 558 | yes |
| `bioclip-2_pollinator-orders-africa_v1` | Africa, at least 3 | 22,358 | 38.0 MB (3.7 MB) | 821 | yes |
| `bioclip-2_pollinator-orders-asia_v1` | Asia, at least 3 | 37,897 | 64.4 MB (6.2 MB) | 963 | yes |
| `bioclip-2_pollinator-orders-oceania_v1` | Oceania, at least 3 | 23,255 | 39.5 MB (3.8 MB) | 67 | yes |

The world list holds every name of the six continent lists (158,125 names) and 34,761 more,
mostly species with 1 or 2 records. Distinct names per country in the world list, alone and
with its neighbours: United States 39,351 and 58,636; Germany 16,937 and 28,534; Australia
16,590 and 17,952; Brazil 10,242 and 19,702; Japan 7,503 and 19,366; South Africa 7,220 and
9,053; India 6,530 and 13,234; Kenya 2,911 and 5,112. Europe v4 against v3: 511 names of v3 are
not in v4 and 551 are new (v3 keeps the rows of v2, counted in October with the older mapping;
GBIF's name matching no longer finds the names with two genera in one row, 72 rows of v3).

BioCLIP 2.5, the same selections (built 2026-11-17; the six continent packs published on Hugging Face on 2026-11-17, the world pack not), with its own mapping
`out/mapping_col26.10xr_bioclip-2.5/` (`make_gbif_mapping.py --model bioclip-2.5`, run on the GPU
workstation with `--workers 4`; 276,384 of 286,485 names with a key), in
`out/packs_2026-11_bioclip-2.5/`. The first BioCLIP 2.5 packs with the `gbif` and `regions` blocks:

| Pack id | Names | Size (header) | Rows in no country list | Opens in the app |
|---|---|---|---|---|
| `bioclip-2.5_pollinator-orders-world_v1` | 189,281 | 418.5 MB (30.8 MB) | 34,418 | no: for PC use |
| `bioclip-2.5_pollinator-orders-europe_v3` | 35,017 | 77.5 MB (5.8 MB) | 457 | yes |
| `bioclip-2.5_pollinator-orders-north-america_v1` | 56,396 | 124.7 MB (9.2 MB) | 407 | yes |
| `bioclip-2.5_pollinator-orders-south-america_v1` | 23,121 | 51.1 MB (3.8 MB) | 540 | yes |
| `bioclip-2.5_pollinator-orders-africa_v1` | 21,765 | 48.1 MB (3.6 MB) | 786 | yes |
| `bioclip-2.5_pollinator-orders-asia_v1` | 37,199 | 82.3 MB (6.1 MB) | 909 | yes |
| `bioclip-2.5_pollinator-orders-oceania_v1` | 22,995 | 50.9 MB (3.8 MB) | 62 | yes |

"Names" in both tables are the names of the species list; a pack can hold a few more rows (one
name can have more than one TreeOfLife row), and the app's catalogue titles and the cards count
rows without the six sink rows (BioCLIP 2 Europe v4: 35,014 names, 35,018 rows).

`build_label_pack.py` exits non-zero after writing a pack the app cannot open (the world pack):
a script that builds several packs must not stop there.

v3 (2026-10-28, on Hugging Face since 2026-11-15): `bioclip-2_pollinator-orders-europe_v3`, the v2 Europe pack
byte for byte (rows, names, vectors) plus the blocks `gbif` (every row has a key: 34,978 rows,
34,974 distinct names, 34,336 distinct keys, synonyms share one) and `regions` (248 countries,
GBIF queried 28 October 2026, presence records, no fossil specimens); 59.5 MB (header 5.8 MB). 636 rows are in no
single country's list: only the whole pack has them. Distinct names per country (at least 3
presence records there) and with its neighbours:

| Country | Alone (its GBIF list for the four orders) | With neighbours |
|---|---|---|
| Germany | 16,766 (16,768) | 28,272 (AT BE CH CZ DK FR LU NL PL SE) |
| France | 17,593 (17,715) | 27,716 |
| Great Britain | 14,708 (14,738) | 22,540 (BE FR GG IE IM) |
| Romania | 3,773 (3,773) | 13,332 |
| Cyprus | 1,024 (1,161) | 3,905 (TR) |
| Malta | 310 (310) | 11,807 (IT) |

The number in brackets is the country's whole list for the four orders: Cyprus shows that some
of its species are not in a Europe pack (Asian species). BioCLIP 2.5 got these blocks with the November
packs above (its own mapping; `add_pack_blocks.py` refuses a species list whose mapping was made
from another TreeOfLife names file).

Mammals and birds v3 (built 2026-11-17, not on Hugging Face; 3b "Mammals and birds"):
`bioclip-2_mammals-birds-world_v3` = v2 byte for byte (rows, names, name rules, vectors) without
the "person" sink row (17,135 rows, 5 sink rows), plus `gbif` (15,526 of 17,130 names with a key,
15,150 distinct keys) and `regions` (250 countries, GBIF queried 2026-11-17, at least 3 presence
records); 29.4 MB (header 3.1 MB). 1,922 rows are in no country list: 1,604 without a key (1,435
mammals, nearly all fossil species of TreeOfLife, e.g. *Astrapotherium magnus*, which GBIF's name
matching does not find), 163 extinct, 155 with under 3 records in every country. Names per
country, alone and with its neighbours: United States 3,063 and 4,247; Brazil 2,598 and 4,623;
Indonesia 2,516 and 3,385; India 1,976 and 2,833; Kenya 1,756 and 2,360; DR Congo 1,609 and
2,459; Germany 1,622 and 2,211; Nigeria 1,241 and 1,569; Malta 298 and 746.

v2 (2026-10-26, `--name-rules higher-rank`, Europe list made the insect-detect-post way):
`bioclip-2_pollinator-orders-europe_v2` (34,978 names), `bioclip-2_mammals-birds-world_v2`
(17,130), `bioclip-2.5_pollinator-orders-europe_v2` (34,392). v1 (2026-09-20 to 10-02, every row
as published):

| Pack id | Selection | Names | Size | Runs in the app today |
|---|---|---|---|---|
| `bioclip-2_flower-visitors-32fam_v1` | 32 flower-visitor families, worldwide (demo, not maintained) | 38,570 | 60 MB | yes |
| `bioclip-2_pollinator-orders-europe_v1` | Diptera, Hymenoptera, Coleoptera, Lepidoptera with GBIF records in Europe | 35,264 | 57 MB | yes |
| `bioclip-2_mammalia-world_v1` | class Mammalia, worldwide, camera-trap sink prompts (`--sink-set mammal`) | 5,999 | 9.4 MB | yes (for MegaDetector "animal" boxes) |
| `bioclip-2_mammals-birds-world_v1` | classes Mammalia and Aves, worldwide, camera-trap sink prompts (round 278; offered for "Mammals and birds" on the home screen) | 17,130 | 28 MB | yes |
| `bioclip-2_pollinator-orders-world_v1` | the four orders, worldwide | 204,620 | 318 MB | not yet (needs the native scorer of a later app round) |
| `bioclip-2.5_pollinator-orders-europe_v1` | as the Europe pack above, for BioCLIP 2.5 (round 264) | 34,704 | 74 MB | yes (with the BioCLIP 2.5 model) |
| `bioclip-2.5_flower-visitors-32fam_v1` | the 32 families, for BioCLIP 2.5 (round 264; demo, not maintained) | 37,461 | 80 MB | yes (with the BioCLIP 2.5 model) |

**Names (`--name-rules`, since 2026-10-25):** packs keep the TreeOfLife names as published;
no name is corrected, built or moved and no row is dropped (the filters also read the names as
published, so the rules never change which rows a pack holds). The few rows that are not in the
plain form are used by the app only down to a higher rank (`--name-rules higher-rank`, the
default; `NAME_RULES` in `build_label_pack.py`):

| Row | Example (TreeOfLife row) | The app uses it down to |
|---|---|---|
| kingdom "Metazoa", which the TreeOfLife-200M card lists as unresolved ("Metazoa instead of Animalia"; NCBI Taxonomy lists Animalia as a synonym of Metazoa) | 515898: Nymphalidae, Pterygota, "Charaxes bernardus Fabricius, 1793" | family, kingdom read as Animalia |
| the species slot starts with another genus than the genus slot | 243250: Kirinia, "Esperarge eversmanni" | family |
| the species slot is not one plain epithet (authors, years, codes, placeholders such as "spec", "indet", "group") | 149186: Bombus, "spec"; 112236: Toxomerus, "CR-11" | genus |

Without the first rule those rows form a separate kingdom branch that the app's top-down
answer never reaches. The app needs no change: it counts a row only down to the rank above its
first empty slot. Each such row is listed in the header (`name_rules.rows`: pack row, TreeOfLife
row, rule, names as published); `source.names_file_sha256` identifies the names file and
`source.species_csv_sha256` the species list. `--name-rules none` uses every row as published,
as the v1 packs above do. Rows the rules touch (Metazoa / two genera / species slot): v2 packs:
BioCLIP 2 Europe 37 / 72 / 0, mammals and birds 41 / 571 / 1, BioCLIP 2.5 Europe none; in the v1
packs: BioCLIP 2 Europe 36 / 74 / 1, 32 families 85 / 309 / 40, BioCLIP 2.5 32 families
81 / 0 / 32.

What the rules do not catch, because they look only at the form of a row: a plain row with a
wrong placement (e.g. three *Notiophilus* ground beetles that also have a row in the weevil
family Brachyceridae), rows with a gap above the genus (343 bird and mammal rows with no order
or no family, which the app then counts only down to class or order), and plain rows of the
BioCLIP 2.5 file that join the genus slot of a BioCLIP 2 two-genera row with its epithet (e.g.
2.5 "Ceratomegilla rufocincta" where the BioCLIP 2 file has Ceratomegilla, "Semiadalia
rufocincta"). A spelling variant of the same genus also counts as two genera (Cheilosia,
"Chilosia picata"), so such a row stops at family.

To cite the names of a pack built with the rules: "TreeOfLife-200M (Imageomics,
https://doi.org/10.57967/hf/8980), names file `embeddings/txt_emb_species.json` (BioCLIP 2;
`txt_emb_bioclip-2.5-vith14.json` for BioCLIP 2.5) with SHA-256 `<names_file_sha256>`, names used
as published. The dataset card lists kingdom Metazoa as unresolved; it is read as Animalia (NCBI
Taxonomy lists Animalia as a synonym). Rows not in the plain form are used only down to genus or
family, as listed in the pack header."

Every pack ends with six "none of these" rows; `--sink-set arthropod` (default) uses
flower-scene prompts, `--sink-set mammal` camera-trap prompts (five since v3 of the mammals pack:
no "person" row; `mammal-v2` is the six-row set of v1 and v2), `--no-sink` none.

Other selections:

```bash
# whole classes (large: ~280k names, ~430 MB; needs the native scorer)
python build_label_pack.py --model bioclip-2 --classes Insecta,Arachnida --out ./out
# whole orders
python build_label_pack.py --model bioclip-2 --orders Diptera,Hymenoptera --out ./out
```

Keep packs under about 100,000 names for the current app version: it scores them in
pure Dart (about 50 ms per crop per 30,000 names) and holds the matrix in memory
(4 bytes per number). Larger packs are built the same way and wait for the native
scorer.

## 4. Verify the export (recommended, about 3 minutes)

```bash
python verify_parity.py --tflite out/bioclip-2_224_fp16.tflite --images /path/to/some/crops \
  --pack out/bioclip-2_flower-visitors-32fam_v1.fpack --limit 50
```

Compares the phone model with the original PyTorch model on your images: cosine
similarity of the embeddings and the same answer at order, family, genus and species with
each pack given (`--pack` takes several). Expected `PARITY OK` with mean cosine >= 0.99 and
family agreement >= 95 %. Each rank is taken the way the app reports it (top-down, a taxon's
probabilities added up); the lines also say how many of the images PyTorch itself is sure
about agree, which tells a quantised file's near-ties apart from real errors (section 2b),
and the largest change of the shown percentage. Any folder of insect photos or crops works
(jpg/png, searched recursively). For crops of whole track IDs, `--crops-csv crops.csv` (one row
per crop: `crop_id`, `kind`, `seq_id`, `crop_px`, `new_filename`) takes one random crop per track
ID and one square per insect-free frame, so long tracks do not fill the sample. The ranks and
the top-down answer come from `app_rules.py`, the Python copy of the app's rules.

## 5. Copy the files to the phone and import them

```bash
adb push out/bioclip-2_224_fp16.tflite /sdcard/Download/
adb push out/bioclip-2_flower-visitors-32fam_v1.fpack /sdcard/Download/
```

(or copy them over USB / a file manager into the phone's Downloads folder). Then in
FaunaPulse: home screen Menu, **Download & import models**, *Import model files…* (the model and the
label pack, both at once: the app sorts them). The app copies both files into its private storage; the
Downloads copies can be deleted afterwards. The app's [`docs/IDENTIFICATION.md`](https://github.com/valentinitnelav/fauna-pulse/blob/main/docs/IDENTIFICATION.md) explains the run and
the results.

## Why the full model on the phone, and not a distilled one (for now)

Distillation (training a small "student" network / model to imitate the BioCLIP image tower - the "teacher")
was considered and is deliberately not used yet.

The reasoning:

- FaunaPulse is aimed to use identification **after** recording, in bulk, on a 
  plugged-in phone (e.g. overnight session). Speed is not the main goal there, accuracy is. 
  The "teacher" (the full BioCLIP 2 image tower, ~600 MiB fp16) is the accuracy ceiling
  and needs no training data.
- Published "students" lose fidelity, mostly at species level: 
  - the [BioCLIP 2.5 to FastViT-SA12 student][nate_distill] of Nate Hamilton (24 MB,
  label-free MobileCLIP-style recipe) agrees with its teacher on 71.7 % of top-1 and
  88.8 % of top-5 answers on plants; 
  - the "ConvNeXt-tiny+KD" student model of [Gardiner et al.][gardiner_2025]
  (ICCVW 2025, 101 moth species) reaches 64.7 % top-1 target accuracy with no 
  field labels against 88.3 % for BioCLIP 2 (Table 1 in manuscript), 
  and needs about 50% data mix with expert-labelled field data to catch up.
  Those authors recommend BioCLIP 2 itself when compute allows and labelled field data
  are scarce, which is mostly FaunaPulse situation (at least for now).

If it becomes a goal, the path is ready because the app is model-agnostic (any `.tflite` that maps
RGB to an embedding in a pack's space, declared in the manifest). So a student model
distilled from **BioCLIP 2** can be used (keeping the 768-dimension text space, so the packs stay
valid). The label-free recipe of [Nate Hamilton][nate_distill] (MIT; cache teacher
embeddings of insect images and of FaunaPulse's own crops, train a FastViT-class
student with a cosine loss) applies unchanged; export with
`export_image_tower.py` (timm models convert the same way) and check agreement with
the teacher on real crops with `verify_parity.py`.
Vasu et al. (2024) MobileCLIP, presents the distillation recipe.

A related, non-distillation idea worth keeping in mind
is a small insect-versus-background head trained on stored embeddings, as in
https://github.com/lollogiro/zero-shot-insect-detection (the app's "none of these"
rows are the zero-shot version of it).

References:

> Gardiner, R. J., Mougeot, G., Rowlands, S., Simmons, B. I., Helsing, F., & Høye, T. T. (2025, October). Bridging Domain Gaps for Fine-Grained Moth Classification Through Expert-Informed Adaptation and Foundation Model Priors. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW) (pp. 5169-5174). IEEE. https://doi.org/10.1109/ICCVW69036.2025.00538;

> Vasu, P. K. A., Pouransari, H., Faghri, F., Vemulapalli, R., & Tuzel, O. (2024, June). Mobileclip: Fast image-text models through multi-modal reinforced training. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 15963-15974). IEEE. https://doi.org/10.1109/CVPR52733.2024.01511

[nate_distill]: https://github.com/CrazedCoderNate/bioclip-mobile-distill
[gardiner_2025]: https://doi.org/10.1109/ICCVW69036.2025.00538

## Troubleshooting

- `ModuleNotFoundError: No module named 'tensorflow'`: an old copy of the script;
  the current one does not use TensorFlow (litert-torch 0.9+ plus ai-edge-quantizer).
- `TypeError: can only concatenate str (not "bytes") to str` inside ai-edge-quantizer:
  a text/bytes mismatch between ai-edge-quantizer 0.9.0 and flatbuffers 25.12.
  `quantise_tflite.py` works around it (tensor names are normalised before the cast).
  If it still appears with newer versions, use `--precision fp32`.
- The fp16 file is larger than the float32 one, or `inspect_tflite.py` reports
  "weights computed at runtime": the conversion ran with `--lightweight`, which leaves
  the LayerNorm-scale products of 48 weight matrices unfolded. Delete the float32 file
  in `out/` and run again without `--lightweight`.
- Out of memory during conversion: close other programs (about 7 GB free RAM is needed
  for BioCLIP 2). `--lightweight` is the last resort (see the previous point).
- Disk: keep about 2 GB free during a run (float32 intermediate plus output).
  `pip cache purge` frees the wheels pip downloaded; the Hugging Face cache
  (`~/.cache/huggingface`) can be deleted after the export (it is re-downloaded on the
  next run).
- Slow PC inference in `verify_parity.py` (about 1 s per crop for fp16, 4 s for
  float32) is normal; it only affects the check, not the phone.

## Files in this folder

| File | Purpose |
|---|---|
| `export_image_tower.py` | checkpoint -> `.tflite` (+ manifest) |
| `quantise_tflite.py` | float32 `.tflite` -> fp16 / int8 / int8-weights, with a quick check (any model) |
| `build_label_pack.py` | TreeOfLife embeddings + filters + sink rows -> `.fpack` |
| `fpack.py` | the pack container format (also writes the app's test fixture) |
| `build_region_species_list.py` | GBIF occurrence facets + TreeOfLife-to-GBIF mapping -> regional species CSV |
| `build_country_lists.py` | one GBIF facet per country -> per-country species keys for `--country-lists` |
| `make_gbif_mapping.py` | runs insect-detect-post's mapping script for the Catalogue of Life release GBIF uses now -> mapping CSV for `--mapping-csv` |
| `add_pack_blocks.py` | an existing pack + its species list (+ country lists) -> the same pack with the `gbif` and `regions` blocks, vectors copied byte for byte |
| `country_neighbours.py` | Natural Earth borders -> `country_neighbours.csv` + `.json` (tracked neighbour table) |
| `check_pack_blocks.py` | checks of a pack's `gbif` and `regions` blocks, the app's header limits, and v3 against v2 |
| `verify_parity.py` | PyTorch vs `.tflite` comparison on real images |
| `app_rules.py` | Python copy of the app's identification defaults and top-down answer (`verify_parity.py`, calibration scripts) |
| `inspect_tflite.py` | ops, constant sizes and how each layer's weights are stored (fp16, int8, float32) |
| `requirements.txt` | packages (version ranges; exact verified versions: `../requirements-lock.txt`) |
| `out/` | outputs (git-ignored) |

## Attribution of methods and data used here

Two sources are credited for different things; please keep both when citing.

**From BioCLIP / pybioclip (Imageomics; model MIT, embeddings CC0, code MIT):**
- zero-shot identification from a hierarchical taxonomic name list, and the
  TreeOfLife-200M name embeddings the packs are cut from (Stevens et al. 2024; Gu et al. 2025);
- the label-subset mechanism the packs implement (pybioclip's `--subset` /
  `create_taxa_filter` / `apply_filter` over precomputed name embeddings);
- the softmax with the model's logit scale and the roll-up of species probabilities to
  higher ranks (pybioclip's `predict` and `format_grouped_probs`);
- the recommendation to restrict candidates to a regional GBIF or Map of Life species
  list ("Geo-Restricted Taxon List Predictions", https://imageomics.github.io/pybioclip/geo-restricted-taxa/).

**From Maximilian Sittinger's `insect-detect-post` (AGPL-3.0; Sittinger, M. 2026, Zenodo
https://doi.org/10.5281/zenodo.21822140), re-implemented in our own code:**
- the API-based way of building the regional list (GBIF occurrence facets, minimum 3
  records) and his TreeOfLife-to-GBIF key mapping (v1.1.0 `bioclip_gbif_species_mapping_Arthropoda.csv`;
  the v1 packs used `tol_gbif_taxon_keys_Arthropoda.csv`; downloaded from his release, not
  redistributed), also for the per-country lists and for the GBIF key of each pack row (his
  `gbif_species`, `gbif_species_key` columns);
- the per-track-ID CSV column names (`pred`, `pred_prob_weighted`, `pred_prob_mean`,
  `track_imgs`, `pred_imgs`, `bioclip_<rank>`) of his `_classified_final.csv`, so both
  tools' outputs can be analysed with the same scripts; the app's pooling rule itself
  differs (round 219: certainty-weighted average of the crops' embeddings scored once, the
  "Average Logit" of Dussert et al. 2025; `reproduce_track_conf.py` in this folder recomputes
  a session's track-id results off the phone from the stored embeddings and a pack, with
  `--countries` also as narrowed to the country lists the app used);
- the square-on-the-longer-side crop rule (`make_bbox_square()`); FaunaPulse pads at
  photo edges instead of shifting the square, and has an optional margin (0 by default
  since round 256).

**Precedent, not origin:** the "none of these" rows follow the common practice of
explicit background classes, as in the `none_*` classes of the Insect Detect
classification dataset (Sittinger, Uhler & Pink 2023, https://doi.org/10.5281/zenodo.8325384).

**Data:** GBIF occurrence facets and backbone taxonomy, https://www.gbif.org; Catalogue of
Life, the release named in each list's `.json` (COL26.6 XR: https://doi.org/10.48580/dgy8b);
country borders for the neighbour table: Natural Earth (public domain), https://www.naturalearthdata.com.

## Licenses and citation

Model weights: BioCLIP 2 and BioCLIP 2.5 by Imageomics, MIT (BioCLIP 2.5 model card:
https://huggingface.co/imageomics/bioclip-2.5-vith14). Name embeddings: TreeOfLife-200M,
CC0-1.0. 

Important note - the converted files are unofficial conversions, not provided or
endorsed by Imageomics.  

If you use these models, even if they were adapted for smartphone usage with FaunaPulse, 
please note the citation suggestions of the BioCLIP2 model card on Hugging Face at
https://huggingface.co/imageomics/bioclip-2#citation

See also the model cards [`hf_cards/bioclip-2-tflite`](../hf_cards/bioclip-2-tflite/README.md)
and [`hf_cards/bioclip-2.5-tflite`](../hf_cards/bioclip-2.5-tflite/README.md).
