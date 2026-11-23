# Datasets

| Dataset | Modality | Region | Name | Num crops | Subvolume size (k actual crop sizes x y z) | Size crop | Voxel size (x x y x z nm) | Voxel count | Task | Split |
|---|---|---|---|---|---|---|---|---|---|---|
| SNEMI3D | ssSEM | Mouse S1 cortex | AC3_em | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC4_em | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC3_cc | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC4_cc | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| Neurons | ssSEM | Mouse S1 cortex | Kasthuri cylinder (cc) | 1 | 01 x 5000 x 2900 x 300 | 5000 x 2900 x 300 | 6 x 6 x 30 | 4,350,000,000 | SFT | Train |
| Neurons | ssSEM | Mouse S1 cortex | Kasthuri cropping (cc) | 1 | 01 x 5904 x 4816 x 1848 | 5904 x 4816 x 1848 | 6 x 6 x 30 | 52,545,411,072 | SSL | Train |
| COSEM3D | FIB-SEM | HeLa cell (jrc_hela-3) | jrc_hela-3_4nm | 6 | 6 x 1659 x 1000 x 2048 (z, y, x) | 2048 x 1000 x 1659 | 4 x 4 x 4 | 3,397,632,000 each | SSL | Train (5) / Val (1) |
| COSEM3D | FIB-SEM | Jurkat cell (jrc_jurkat-1) | jrc_jurkat-1_4nm | 3 | 3 x 1761 x 2048 x 2048 (z, y, x) | 2048 x 2048 x 1761 | 4 x 4 x 4 | 7,386,169,344 each | SSL | Train (2) / Val (1) |
| COSEM3D | FIB-SEM | Macrophage (jrc_macrophage-2) | jrc_macrophage-2_4nm | 2 | 2 x 1720 x 2000 x 2048 (z, y, x) | 2048 x 2000 x 1720 | 4 x 4 x 4 | 7,045,120,000 each | SSL | Train (1) / Val (1) |

Files live in `data/SNEMI3D/` (not tracked by git), HDF5 key `main`, array order (z, y, x), uint8
images. Image files are tagged by contrast: `_em_` = uncorrected intensities from BossDB
`Kasthuri/em/images` (slice-mean brightness swings ~20 grey levels between neighbouring slices);
`_cc_` = contrast-corrected `kasthuri2015/em/cc` (flat per-slice brightness). Label files carry no
tag and pair with both versions. Coordinates are 6 nm voxels in the Kasthuri et al. (2015) volume as
served by BossDB; z is the slice index of the named layer.

| Name | Image | Labels |
|---|---|---|
| AC3_em | `AC3_em_inputs.h5` | `AC3_labels.h5` |
| AC4_em | `AC4_em_inputs.h5` | `AC4_labels.h5` |
| AC3_cc | `AC3_cc_inputs.h5` | `AC3_labels.h5` |
| AC4_cc | `AC4_cc_inputs.h5` | `AC4_labels.h5` |
| SNEMI3D train / test | `train_em_inputs.h5` / `test_em_inputs.h5` | `train_labels.h5` / none |
| Kasthuri cylinder (cc) | `neurons_cc_5000x2900x300_x3000_y7200_z950_volume.h5` | `neurons_5000x2900x300_x3000_y7200_z950_segmentation.h5` |
| Kasthuri cropping (cc) | `neurons_cc_5904x4816x1848_x2512_y4432_z0_volume.h5` | none |

`images` and `cc` are the same sections with the same x/y, but `cc` z = `images` z + 2 (the paper's
AC3/AC4 bounding boxes use the `cc` numbering). Within that mapping the residual offset is under one
6 nm voxel (the two layers were downsampled on grids about half a voxel apart).

- **AC3**: x 5472-6496, y 8712-9736, `images` z 1153-1252 / `cc` z 1155-1254 (the top 100 of AC3's
  256 slices), stored in reverse slice order. `AC3_em` is identical to SNEMI3D `test_inputs.h5`; its
  slice 89 is a near-copy of slice 90 rather than the real section. `AC3_cc` slice 89 is the real
  section, and the labels fit it better.
- **AC4**: x 4400-5424, y 5440-6464, `images` z 1097-1196 / `cc` z 1099-1198, stored with z, y and x
  all reversed. `AC4_em` is identical to BossDB `Kasthuri/ac4/em` (same flips); labels equal
  `Kasthuri/ac4/neuron` up to a one-to-one relabel.
- **SNEMI3D train / test**: the original challenge files. `test_em_inputs.h5` is identical to
  `AC3_em_inputs.h5`. `train_em_inputs.h5` equals `AC4_em_inputs.h5` except slice 33 (a near-copy of
  slice 34), and `train_labels.h5` is `AC4_labels.h5` with wider background borders (15 % vs 10 %).
- **Kasthuri cylinder**: offset (x, y, z) = (3000, 7200, 950) in `kasthuri2015/em/cc`. Labels are a
  crop of `kasthuri2015/em/3cylneuron_v1`; only ~26 % of voxels are labelled and 55 of 300 slices
  have none.
- **Kasthuri cropping**: largest crop with tissue in every voxel across all slices of
  `kasthuri2015/em/cc`: x 2512-8416, y 4432-9248, z 0-1848. Raw only, stored uncompressed with
  (16, 256, 256) chunks. Slice 432 has a section defect (a ~0.6 % black blob near x 4450-5070,
  y 340-975 in crop coordinates).

**COSEM3D** (`data/COSEM3D/`, HDF5 key `main`, array order (z, y, x), uint8, each with a `.norm.json` sidecar
holding min 0 / max 255): only the true 4 x 4 x 4 nm cubic files (`*_4nm_*`) are copied from
`/home/nvidia/data/COSEM3D`. They were resampled to cubic voxels from the 4 x 4 x 3.24 nm (HeLa) and
4 x 4 x 3.44 nm (Jurkat) originals; those non-cubic files are not copied. The suffix `x<X>_y<Y>_z<Z>` is
the tile's origin in the source volume. Raw only, no labels. COSEM3D is currently **not** in `configs/pretrain_snemi3d.yaml`. The tiles below
were in an earlier version of the config (the train tiles were
HeLa x4096_z2048 / x6144_z2048 / x4096_z0 / x4096_z4096 / x2048_z4096, Macrophage x4096_z4096 and Jurkat
x4096_z0 / x4096_z4096; the validation tiles are HeLa x6144_z0, Macrophage x2048_z4096 and Jurkat
x0_z2048).
