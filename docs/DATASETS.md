# Datasets

| Dataset | Modality | Region | Name | Num crops | Subvolume size (k actual crop sizes x y z) | Size crop | Voxel size (x x y x z nm) | Voxel count | Task | Split |
|---|---|---|---|---|---|---|---|---|---|---|
| SNEMI3D | ssSEM | Mouse S1 cortex | AC3_em | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC3_cc | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC4_em | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SSL | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | AC4_cc | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SSL | Test |
| SNEMI3D | ssSEM | Mouse S1 cortex | SNEMI3D train | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SFT | Train |
| SNEMI3D | ssSEM | Mouse S1 cortex | SNEMI3D test | 1 | 01 x 1024 x 1024 x 100 | 1024 x 1024 x 100 | 6 x 6 x 30 | 104,857,600 | SSL | Test |
| Neurons | ssSEM | Mouse S1 cortex | Kasthuri cylinder (cc) | 1 | 01 x 5000 x 2900 x 300 | 5000 x 2900 x 300 | 6 x 6 x 30 | 4,350,000,000 | SFT | Train |
| Neurons | ssSEM | Mouse S1 cortex | Kasthuri cropping (cc) | 1 | 01 x 5904 x 4816 x 1848 | 5904 x 4816 x 1848 | 6 x 6 x 30 | 52,545,411,072 | SSL | Train |

Files live in `data/SNEMI3D/` (not tracked by git), HDF5 key `main`, array order (z, y, x), uint8
images. Image files are tagged by contrast: `_em_` = uncorrected intensities from BossDB
`Kasthuri/em/images` (slice-mean brightness steps by a median ~15-17 grey levels between neighbouring slices, max ~80);
`_cc_` = contrast-corrected `kasthuri2015/em/cc` (flat per-slice brightness). Label files carry no
tag and pair with both versions. Coordinates are 6 nm voxels in the Kasthuri et al. (2015) volume as
served by BossDB; z is the slice index of the named layer. x and y ranges below are half-open
`[start, end)`; the AC3/AC4 z ranges list the first and last slice (inclusive), the cylinder and
cropping z are `[start, start + depth)`. `notebooks/data_SNEMI3D.ipynb` re-checks this page against
the files.

| Name | Image | Labels |
|---|---|---|
| AC3_em | `AC3_em_inputs.h5` | `AC3_labels.h5` |
| AC4_em | `AC4_em_inputs.h5` | `AC4_labels.h5` |
| AC3_cc | `AC3_cc_inputs.h5` | `AC3_labels.h5` |
| AC4_cc | `AC4_cc_inputs.h5` | `AC4_labels.h5` |
| SNEMI3D train | `train_em_inputs.h5` | `train_labels.h5` |
| SNEMI3D test | `test_em_inputs.h5` | none |
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
  crop of `kasthuri2015/em/3cylneuron_v1`; labels cover ~42 % of voxels (8-64 % per slice) and every
  slice has some. The image matches `cc` at that offset with NCC 0.995 (mean |diff| 2.6 grey levels), not byte-for-byte;
  its `.norm.json` max is 251, but the data reaches 255, so the norm does not cover values 252-255.
- **Kasthuri cropping**: largest crop with tissue in every voxel across all slices of
  `kasthuri2015/em/cc`: x 2512-8416, y 4432-9248, z 0-1848. Raw only, stored uncompressed with
  (16, 256, 256) chunks. Slice 432 has a section defect (a ~0.6 % black blob near x 4450-5070,
  y 580-975 in crop coordinates).
