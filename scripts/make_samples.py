#!/usr/bin/env python
"""Copy the teaching crops out of the nanoCosmos data tree into ./em_samples (numpy + h5py only, CPU, deterministic).

    python make_em_samples.py [--data-root /home/nvidia/nanoCosmos/data] [--out ./em_samples] [--only snemi3d cremi_a fib25]

The data tree is opened read-only. One HDF5 per sample, axis order (z, y, x):
  /raw                    uint8, gzip 4            EM intensity, source values untouched
  /instance               uint8 | uint16 | uint32  compact ids 0..K, 0 = background, 1 = the LARGEST instance (absent if image-only)
  /instance_original_ids  uint64 (K+1,)            original id of compact id i
  /instance_voxel_counts  uint32 (K+1,)            voxels per compact id (entry 0 = background)
  /instance_bbox_zyx      int32 (K+1, 6)           z0, y0, x0, z1, y1, x1, upper bounds exclusive (row 0 unused)
  root attributes         voxel_size_zyx_nm, axis_order, source_*, intensity_norm_*, label_convention, citation, licence, ...
"""
import argparse, json, os
import h5py
import numpy as np

CROPS = json.loads(r"""[
 {
  "name": "snemi3d",
  "role": "core",
  "raw": "SNEMI3D/AC3_inputs.h5",
  "seg": "SNEMI3D/AC3_labels.h5",
  "offset_zyx": [
   0,
   320,
   416
  ],
  "shape_zyx": [
   100,
   512,
   512
  ],
  "voxel_size_zyx_nm": [
   30.0,
   6.0,
   6.0
  ],
  "label_convention": "gapped",
  "skeleton_subject_ids": [
   10,
   2,
   5
  ],
  "expected_n_instances": 138,
  "expected_bytes": 25276617,
  "dataset_name": "SNEMI3D, stack AC3 (the challenge test stack; image identical to test_inputs.h5)",
  "species": "mouse (Mus musculus)",
  "tissue": "somatosensory cortex (S1)",
  "imaging_modality": "ssSEM (ATUM tape collection + SEM)",
  "citation": "Kasthuri, N. et al. (2015), Saturated Reconstruction of a Volume of Neocortex, Cell 162(3):648-661.",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "AC3_labels.h5 is not produced by any nanoCosmos script and its provenance is undocumented locally; it is aligned with the image (membranes are darker under label 0). Section brightness flickers strongly (per-section mean 110..207): normalise per section before volume rendering."
 },
 {
  "name": "cremi_a",
  "role": "core",
  "raw": "CREMI3D/cremi3d_sample_A_volume.h5",
  "seg": "CREMI3D/cremi3d_sample_A_segmentation.h5",
  "offset_zyx": [
   45,
   120,
   120
  ],
  "shape_zyx": [
   75,
   768,
   768
  ],
  "voxel_size_zyx_nm": [
   40.0,
   4.0,
   4.0
  ],
  "label_convention": "filled",
  "skeleton_subject_ids": [
   30,
   5,
   2
  ],
  "expected_n_instances": 9017,
  "expected_bytes": 37380680,
  "dataset_name": "CREMI challenge (MICCAI 2016), sample A",
  "species": "fruit fly (Drosophila melanogaster), adult",
  "tissue": "brain",
  "imaging_modality": "ssTEM",
  "citation": "Funke, Saalfeld, Bock, Turaga, Perlman - CREMI Challenge, https://cremi.org/ ; FAFB: Zheng et al. 2018, Cell.",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Ids tile space (no zeros). Thousands of ids are specks of a few voxels: ids are size-ordered, so keep ids <= N for the large neurites. Almost every neurite runs across the sections (along z)."
 },
 {
  "name": "fib25",
  "role": "core",
  "raw": "FLYEM3D/flyem3d_8nm_x2304_y2048_z6144_volume.h5",
  "seg": "FLYEM3D/flyem3d_8nm_x2304_y2048_z6144_segmentation.h5",
  "offset_zyx": [
   972,
   180,
   145
  ],
  "shape_zyx": [
   384,
   384,
   384
  ],
  "voxel_size_zyx_nm": [
   8.0,
   8.0,
   8.0
  ],
  "label_convention": "filled",
  "skeleton_subject_ids": [
   25,
   23,
   10
  ],
  "expected_n_instances": 768,
  "expected_bytes": 52045338,
  "dataset_name": "FlyEM FIB-25 (7-column medulla), dense core crop x2304 y2048 z6144",
  "species": "fruit fly (Drosophila melanogaster)",
  "tissue": "optic lobe, medulla",
  "imaging_modality": "FIB-SEM",
  "citation": "Takemura, S. et al. (2015), PNAS 112(44):13711-13716, doi:10.1073/pnas.1509820112.",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Isotropic reference. raw[p::4] and instance[p::4] reproduce the 32x8x8 nm FIB253D twin exactly (see manifest 'derived'). The crop lies inside one 500-voxel block of the ground truth, so no label seam crosses it."
 },
 {
  "name": "flywire",
  "role": "extra",
  "raw": "FLYWIRE/flywire_mip1_1024x1024x256_x88000_y20000_z3600_volume.h5",
  "seg": "FLYWIRE/flywire_mip1_1024x1024x256_x88000_y20000_z3600_m783_segmentation.h5",
  "offset_zyx": [
   181,
   160,
   640
  ],
  "shape_zyx": [
   75,
   384,
   384
  ],
  "voxel_size_zyx_nm": [
   40.0,
   8.0,
   8.0
  ],
  "label_convention": "filled",
  "skeleton_subject_ids": [
   29,
   17,
   3
  ],
  "expected_n_instances": 620,
  "expected_bytes": 9674056,
  "dataset_name": "FlyWire FAFB, materialization v783 (EM mip 1, 8 nm; segmentation 16 nm nearest-upsampled 2x in xy)",
  "species": "fruit fly (Drosophila melanogaster), adult female",
  "tissue": "brain (FAFB)",
  "imaging_modality": "ssTEM",
  "citation": "FlyWire Consortium (2024), Whole-brain annotation and multi-connectome cell typing of Drosophila, Nature, doi:10.1038/s41586-024-07686-5; Dorkenwald et al. 2024, Nature.",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Labels are 16 nm data upsampled 2x in xy, so label edges move in 2-voxel steps in plane."
 },
 {
  "name": "microns",
  "role": "extra",
  "raw": "MICRONS/minnie65_mip0_4096x4096x800_x80000_y70000_z16500_volume.h5",
  "seg": "MICRONS/minnie65_mip0_4096x4096x800_x80000_y70000_z16500_v1300_segmentation.h5",
  "offset_zyx": [
   650,
   300,
   1000
  ],
  "shape_zyx": [
   75,
   384,
   384
  ],
  "voxel_size_zyx_nm": [
   40.0,
   8.0,
   8.0
  ],
  "label_convention": "filled",
  "skeleton_subject_ids": [
   31,
   12,
   4
  ],
  "expected_n_instances": 278,
  "expected_bytes": 7835703,
  "dataset_name": "MICrONS minnie65, EM mip 0, segmentation v1300",
  "species": "mouse (Mus musculus)",
  "tissue": "visual cortex (V1)",
  "imaging_modality": "ssTEM",
  "citation": "MICrONS Consortium (2025), Functional connectomics spanning multiple areas of mouse visual cortex, Nature (preprint bioRxiv 2021.07.28.454025).",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Original ids are 18-digit uint64 root ids: keep them as integers or strings, never as float or JSON numbers. Low native contrast: display with the intensity_norm window. The source files carry no HDF5 attributes; the voxel size comes from the docs."
 },
 {
  "name": "h01",
  "role": "extra",
  "raw": "H01Cell/h01cell_mip0_1024x1024x5000_x560000_y200000_z0_volume.h5",
  "seg": "H01Cell/h01cell_mip0_1024x1024x5000_x560000_y200000_z0_c3_segmentation.h5",
  "offset_zyx": [
   2046,
   0,
   480
  ],
  "shape_zyx": [
   47,
   384,
   384
  ],
  "voxel_size_zyx_nm": [
   33.0,
   4.0,
   4.0
  ],
  "label_convention": "gapped",
  "skeleton_subject_ids": [
   6,
   7,
   4
  ],
  "expected_n_instances": 68,
  "expected_bytes": 7122735,
  "dataset_name": "H01 release (human cortex), EM mip 0 (4 nm), c3 segmentation (8 nm nearest-upsampled 2x in xy)",
  "species": "human (Homo sapiens)",
  "tissue": "temporal cortex",
  "imaging_modality": "ssSEM (ATUM tape collection + multibeam SEM, per the H01 paper; the local nanoCosmos docs say ssTEM)",
  "citation": "Shapson-Coe et al. 2024, Science (A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution).",
  "licence": "CC-BY-4.0 (segmentation, as stated in scripts/download_h01cell.py); EM: not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Half-size field of view (about 1.5 um). Stain is saturated (many 0 and 255 voxels). About a quarter of the voxels are unlabelled."
 },
 {
  "name": "mitoem2_podo",
  "role": "extra",
  "raw": "MitoEM2/mitoem2_podo_train01_volume.h5",
  "seg": null,
  "offset_zyx": [
   0,
   576,
   480
  ],
  "shape_zyx": [
   192,
   192,
   192
  ],
  "voxel_size_zyx_nm": [
   16.0,
   16.0,
   16.0
  ],
  "label_convention": "none",
  "skeleton_subject_ids": [],
  "expected_n_instances": 0,
  "expected_bytes": 5696379,
  "dataset_name": "MitoEM 2.0 v6, Dataset005_ME2-Podo, train01 (image only; mitochondria labels are not on disk)",
  "species": "mouse (Mus musculus)",
  "tissue": "kidney (glomerulus), podocyte",
  "imaging_modality": "FIB-SEM",
  "citation": "Liu, P. (2026), MitoEM 2.0 v6, Zenodo, doi:10.5281/zenodo.20417683; source data doi:10.1038/s41586-021-03977-3 (MitoEM2 metadata.csv).",
  "licence": "CC-BY-4.0 (MitoEM2 metadata.csv)",
  "notes": "Isotropic but coarse: the same 3 um cube needs only 192^3 voxels."
 },
 {
  "name": "cosem_hela3",
  "role": "extra",
  "raw": "COSEM3D/jrc_hela-3_4nm_x6144_y0_z0_volume.h5",
  "seg": null,
  "offset_zyx": [
   1056,
   528,
   1312
  ],
  "shape_zyx": [
   192,
   192,
   192
  ],
  "voxel_size_zyx_nm": [
   4.0,
   4.0,
   4.0
  ],
  "label_convention": "none",
  "skeleton_subject_ids": [],
  "expected_n_instances": 0,
  "expected_bytes": 6743858,
  "dataset_name": "OpenOrganelle / COSEM jrc_hela-3 (image only), crop x6144 y0 z0",
  "species": "human (Homo sapiens), HeLa cell line",
  "tissue": "cultured cell",
  "imaging_modality": "FIB-SEM",
  "citation": "Xu et al. 2021, Nature; Heinrich et al. 2021, Nature.",
  "licence": "not stated in the local docs (nanoCosmos doc/DATASETS.md, scripts/download_*.py)",
  "notes": "Quarter-size field of view (0.768 um; the full 3.07 um cube would need 768^3 = 453 M voxels). On disk the z axis was resampled 3.24 -> 4 nm (trilinear) by nanoCosmos resample_cosem_isotropic.py, so 4 nm in z is a resampled grid, not the native sampling."
 }
]""")


def compact_by_size(lab):
    """0 stays background; compact id 1 is the largest instance (ties: smaller original id first)."""
    uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    inv = inv.reshape(lab.shape)
    fg = np.flatnonzero(uniq != 0)
    order = fg[np.lexsort((uniq[fg], -cnt[fg]))]
    lut = np.zeros(uniq.size, dtype=np.uint32)
    lut[order] = np.arange(1, order.size + 1, dtype=np.uint32)
    k = order.size
    dtype = np.uint8 if k <= 0xFF else np.uint16 if k <= 0xFFFF else np.uint32
    orig = np.zeros(k + 1, dtype=np.uint64)
    orig[1:] = uniq[order].astype(np.uint64)
    counts = np.zeros(k + 1, dtype=np.uint32)
    counts[1:] = cnt[order]
    counts[0] = lab.size - int(cnt[order].sum())
    return lut[inv].astype(dtype), orig, counts


def bboxes(compact, k):
    """(K+1, 6) int32: z0, y0, x0, z1, y1, x1 (exclusive upper bounds); row 0 (background) is all zeros."""
    out = np.zeros((k + 1, 6), dtype=np.int32)
    flat = compact.ravel()
    for ax in range(3):
        shape = [1, 1, 1]
        shape[ax] = compact.shape[ax]
        coord = np.broadcast_to(np.arange(compact.shape[ax], dtype=np.int32).reshape(shape), compact.shape).ravel()
        lo = np.full(k + 1, compact.shape[ax], dtype=np.int32)
        hi = np.full(k + 1, -1, dtype=np.int32)
        np.minimum.at(lo, flat, coord)
        np.maximum.at(hi, flat, coord)
        out[:, ax], out[:, ax + 3] = lo, hi + 1
    out[0] = 0
    return out


def write_sample(c, data_root, out_dir):
    voxel = np.asarray(c["voxel_size_zyx_nm"], dtype=np.float64)
    sl = tuple(slice(o, o + s) for o, s in zip(c["offset_zyx"], c["shape_zyx"]))
    raw_path = os.path.join(data_root, c["raw"])
    with h5py.File(raw_path, "r") as f:               # read-only
        full_shape = f["main"].shape
        raw = f["main"][sl]
    assert raw.dtype == np.uint8
    sidecar = raw_path + ".norm.json"
    norm = json.load(open(sidecar)) if os.path.exists(sidecar) else None
    compact = None
    if c["seg"]:
        with h5py.File(os.path.join(data_root, c["seg"]), "r") as f:
            lab = f["main"][sl]
        compact, orig, counts = compact_by_size(lab)
        assert (orig[compact] == lab.astype(np.uint64)).all(), "ids must stay traceable"
        assert orig.size - 1 == c["expected_n_instances"], (c["name"], orig.size - 1)
    chunks = tuple(min(a, b) for a, b in zip((16, 128, 128) if voxel[0] / voxel[2] >= 2 else (64, 64, 64), raw.shape))
    kw = dict(compression="gzip", compression_opts=4, shuffle=False, track_times=False)
    path = os.path.join(out_dir, c["name"] + ".h5")
    with h5py.File(path, "w") as f:
        for name, arr in (("raw", raw), ("instance", compact)):
            if arr is None:
                continue
            d = f.create_dataset(name, data=arr, chunks=chunks, **kw)
            d.attrs["voxel_size_zyx_nm"] = voxel
            d.attrs["resolution_zyx_nm"] = voxel
            d.attrs["axis_order"] = "zyx"
        if compact is not None:
            f.create_dataset("instance_original_ids", data=orig, track_times=False)
            f.create_dataset("instance_voxel_counts", data=counts, track_times=False)
            f.create_dataset("instance_bbox_zyx", data=bboxes(compact, orig.size - 1), track_times=False)
        a = f.attrs
        a["layout_version"] = 1
        a["sample"] = c["name"]
        a["dataset_name"] = c["dataset_name"]
        a["voxel_size_zyx_nm"] = voxel
        a["axis_order"] = "zyx"
        a["physical_extent_zyx_nm"] = voxel * np.asarray(raw.shape)
        a["source_path"] = raw_path
        a["source_dataset"] = "main"
        a["source_label_path"] = os.path.join(data_root, c["seg"]) if c["seg"] else ""
        a["source_label_dataset"] = "main" if c["seg"] else ""
        a["source_offset_zyx"] = np.asarray(c["offset_zyx"], dtype=np.int64)
        a["source_shape_zyx"] = np.asarray(c["shape_zyx"], dtype=np.int64)
        a["source_full_shape_zyx"] = np.asarray(full_shape, dtype=np.int64)
        a["intensity_norm_min"] = float(norm["min"]) if norm else float("nan")
        a["intensity_norm_max"] = float(norm["max"]) if norm else float("nan")
        a["intensity_norm_source"] = os.path.basename(sidecar) if norm else "no sidecar next to the source file"
        p01, p99 = np.percentile(raw, [1, 99])
        a["intensity_p01"], a["intensity_p99"] = float(p01), float(p99)
        a["label_convention"] = c["label_convention"]
        a["instance_id_order"] = "compact id 1 is the largest instance (descending voxel count, ties by ascending original id)" if c["seg"] else ""
        a["n_instances"] = int(orig.size - 1) if compact is not None else 0
        a["skeleton_subject_ids"] = np.asarray(c["skeleton_subject_ids"], dtype=np.int64)
        for key in ("species", "tissue", "imaging_modality", "citation", "licence", "notes"):
            a[key] = c[key]
    return path, os.path.getsize(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/nvidia/nanoCosmos/data")
    ap.add_argument("--out", default="./em_samples")
    ap.add_argument("--only", nargs="*", default=None, help="sample names; default: all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total, rows = 0, []
    for c in CROPS:
        if args.only and c["name"] not in args.only:
            continue
        path, size = write_sample(c, args.data_root, args.out)
        total += size
        rows.append(dict(name=c["name"], file=os.path.basename(path), role=c["role"], bytes=size, voxel_size_zyx_nm=c["voxel_size_zyx_nm"], shape_zyx=c["shape_zyx"],
                         source_raw=c["raw"], source_label=c["seg"], source_offset_zyx=c["offset_zyx"], label_convention=c["label_convention"],
                         skeleton_subject_ids=c["skeleton_subject_ids"]))
        print(f"{os.path.basename(path):18s} {size / 1e6:7.2f} MB  (expected {c['expected_bytes'] / 1e6:.2f})")
    json.dump(dict(layout_version=1, axis_order="zyx", total_bytes=total, samples=rows,
                   derived=[dict(name="fib25_z32", derived_from="fib25", recipe="raw[p::4], instance[p::4], p in 0..3", voxel_size_zyx_nm=[32.0, 8.0, 8.0])]),
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"total {total / 1e6:.2f} MB -> {args.out}")
