"""Tests of ncntm.viz.samples on the shipped sample files.

    python -m pytest tests/test_samples.py        or        python tests/test_samples.py

The samples directory is $NCNTM_SAMPLES if set, else <repository>/samples.
"""
import os
import sys
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ncntm.viz import samples as S  # noqa: E402
from ncntm.viz.grid import VoxelGrid  # noqa: E402


@lru_cache(maxsize=None)
def sample(name):
    """Each file is read once for the whole test run."""
    return S.load_sample(name)


def raises(error, fn, *args, **kwargs):
    """True if fn(*args, **kwargs) raises ``error`` (so the tests also run without pytest)."""
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


# ---- where the files are ---------------------------------------------------------------
def test_samples_dir_is_the_repository_folder_or_the_environment_variable():
    saved = os.environ.pop("NCNTM_SAMPLES", None)
    try:
        assert S.samples_dir() == REPO / "samples"
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NCNTM_SAMPLES"] = tmp
            assert S.samples_dir() == Path(tmp).resolve()
            assert raises(FileNotFoundError, S.manifest)          # an empty folder has no manifest
            del os.environ["NCNTM_SAMPLES"]
    finally:
        if saved is not None:
            os.environ["NCNTM_SAMPLES"] = saved


def test_list_samples_follows_the_manifest():
    names = S.list_samples()
    assert names == [entry["name"] for entry in S.manifest()["samples"]]
    assert names[:3] == ["snemi3d", "cremi_a", "fib25"]            # the core samples come first
    assert len(names) == len(set(names))
    for entry in S.manifest()["samples"]:
        assert (S.samples_dir() / entry["file"]).is_file(), entry["file"]


def test_loading_does_not_depend_on_the_working_directory_and_leaves_the_file_alone():
    path = S.samples_dir() / "h01.h5"
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            s = S.load_sample("h01")
        finally:
            os.chdir(here)
    assert s.path == path and s.raw.shape == (47, 384, 384)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == before
    assert raises(KeyError, S.load_sample, "no_such_sample")


# ---- rawvalue, voxvalue, grid ------------------------------------------------------------
def test_snemi3d_arrays_grid_and_attributes():
    s = sample("snemi3d")
    assert s.name == "snemi3d" and s.path.name == "snemi3d.h5"
    assert s.raw.dtype == np.uint8 and s.raw.shape == (100, 512, 512)
    assert s.rawvalue is s.raw                                    # the stored array itself
    assert s.instance.shape == s.raw.shape and s.instance.dtype == np.uint8
    assert s.has_instances
    assert isinstance(s.grid, VoxelGrid)
    assert s.grid.shape_zyx == (100, 512, 512)
    assert s.grid.voxel_size_zyx_nm == (30.0, 6.0, 6.0)
    assert np.allclose(s.grid.extent_xyz_nm, [3072.0, 3072.0, 3000.0])
    assert s.grid.anisotropy == 5.0
    assert s.norm_window == (s.attrs["intensity_norm_min"], s.attrs["intensity_norm_max"]) == (0.0, 255.0)
    assert s.attrs["voxel_size_zyx_nm"] == [30.0, 6.0, 6.0]       # plain Python values
    assert s.attrs["label_convention"] == "gapped"
    assert np.array_equal(s.origin_xyz_nm, [0.0, 0.0, 0.0]) and s.derivation == ()
    assert "snemi3d" in repr(s) and "30 nm" in repr(s)


def test_voxvalue_is_the_clipped_window_and_leaves_rawvalue_untouched():
    # cremi_a has grey values on both sides of its window (20, 199), so the clip is exercised
    c = sample("cremi_a")
    lo, hi = c.norm_window
    assert (lo, hi) == (20.0, 199.0)
    assert c.raw.min() < lo and c.raw.max() > hi
    checksum = int(c.raw.sum(dtype=np.int64))
    v = c.voxvalue()
    assert v.dtype == np.float32 and v.shape == c.raw.shape
    assert v.min() == 0.0 and v.max() == 1.0
    expected = np.clip((c.raw[::7, ::5, ::3].astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    assert np.allclose(v[::7, ::5, ::3], expected, atol=1e-6)
    assert (v[c.raw <= lo] == 0.0).all() and (v[c.raw >= hi] == 1.0).all()
    inside = (c.raw > lo) & (c.raw < hi)
    assert (v[inside] > 0.0).all() and (v[inside] < 1.0).all()
    assert c.raw.dtype == np.uint8 and int(c.raw.sum(dtype=np.int64)) == checksum   # rawvalue untouched


def test_equalize_sections_removes_the_snemi3d_flicker():
    s = sample("snemi3d")
    section_means = s.raw.mean(axis=(1, 2))                       # one number per z section
    assert 110.0 <= section_means.min() < 111.0 and 207.0 <= section_means.max() < 208.0
    plain = s.voxvalue()
    flat = s.voxvalue(equalize_sections=True)
    assert flat.dtype == np.float32 and flat.min() >= 0.0 and flat.max() <= 1.0
    # spread of the section means, in grey levels: 22.5 before, under 0.5 after
    before = float((plain.mean(axis=(1, 2)) * 255.0).std())
    after = float((flat.mean(axis=(1, 2)) * 255.0).std())
    assert 20.0 < before < 25.0, before
    assert after < 0.5, after


# ---- instances -----------------------------------------------------------------------------
def test_instance_tables_agree_with_the_label_array():
    s = sample("snemi3d")
    ids = s.instance_ids()
    assert np.array_equal(ids, np.arange(1, s.attrs["n_instances"] + 1))       # 1 = largest
    counts = [s.voxel_count(i) for i in ids]
    assert counts == sorted(counts, reverse=True)
    for i in (1, 10, 77, int(ids[-1])):
        where = np.argwhere(s.instance == i)
        assert s.voxel_count(i) == len(where)
        assert s.bbox_zyx(i) == tuple(where.min(axis=0)) + tuple(where.max(axis=0) + 1)
    assert raises(ValueError, s.voxel_count, 0)                   # background is not an instance
    assert raises(ValueError, s.bbox_zyx, 9999)


def test_mask_is_boolean_full_size_and_one_piece_by_default():
    from scipy import ndimage

    f = sample("fib25")
    i = f.skeleton_subjects[2]                                    # an id that has several pieces
    everything = f.mask(i, largest_component=False)
    assert everything.dtype == bool and everything.shape == f.raw.shape
    assert everything.sum() == f.voxel_count(i)
    assert np.array_equal(everything, f.instance == i)
    piece = f.mask(i)
    assert piece.dtype == bool and piece.shape == f.raw.shape
    assert (piece <= everything).all()                            # a subset
    assert 0.9 * everything.sum() < piece.sum() < everything.sum()
    labelled, n = ndimage.label(piece)                            # 6-connectivity
    assert n == 1
    _, n_all = ndimage.label(everything)
    assert n_all > 1
    sizes = np.bincount(ndimage.label(everything)[0].ravel())[1:]
    assert piece.sum() == sizes.max()                             # it is the largest one


def test_skeleton_subjects_come_from_the_file():
    assert sample("snemi3d").skeleton_subjects == (10, 2, 5)
    assert sample("fib25").skeleton_subjects == (25, 23, 10)
    assert sample("snemi3d").skeleton_subjects == tuple(sample("snemi3d").attrs["skeleton_subject_ids"])


def test_image_only_sample_says_so():
    c = sample("cosem_hela3")
    assert not c.has_instances and c.instance is None
    assert c.skeleton_subjects == ()
    assert c.voxvalue().shape == c.raw.shape
    assert raises(ValueError, c.mask, 1)
    assert raises(ValueError, c.instance_ids)
    assert "image only" in c.describe()


# ---- strided: the controlled anisotropy experiment ---------------------------------------------
def test_fib25_strided_by_4_is_the_32x8x8_volume():
    f = sample("fib25")
    assert f.grid.voxel_size_zyx_nm == (8.0, 8.0, 8.0) and f.grid.is_isotropic
    g = f.strided(4)
    assert g.grid.voxel_size_zyx_nm == (32.0, 8.0, 8.0)
    assert g.grid.shape_zyx == (96, 384, 384) == g.raw.shape == g.instance.shape
    assert g.grid == f.grid.strided(4, 0)
    assert np.allclose(g.grid.extent_xyz_nm, f.grid.extent_xyz_nm)          # same piece of tissue
    assert g.grid.anisotropy == 4.0
    assert np.array_equal(g.raw, f.raw[0::4]) and np.array_equal(g.instance, f.instance[0::4])
    assert g.raw.dtype == np.uint8 and g.instance.dtype == f.instance.dtype
    assert g.norm_window == f.norm_window and g.skeleton_subjects == f.skeleton_subjects
    assert "strided" in g.derivation[-1] and "32 nm" in g.describe()
    # tables are recomputed for the strided labels
    i = 25
    assert g.voxel_count(i) == int((f.instance[0::4] == i).sum())
    assert g.mask(i, largest_component=False).shape == (96, 384, 384)

    p = f.strided(4, phase=2)
    assert np.array_equal(p.raw, f.raw[2::4]) and np.array_equal(p.instance, f.instance[2::4])


def test_slab_mean_averages_the_image_but_never_the_labels():
    f = sample("fib25").crop(slice(0, 64), slice(0, 96), slice(0, 96))
    g = f.strided(4, phase=1, mode="slab_mean")
    expected = np.rint(f.raw.reshape(16, 4, 96, 96).mean(axis=1)).astype(np.uint8)
    assert g.raw.dtype == np.uint8 and np.array_equal(g.raw, expected)
    assert np.array_equal(g.instance, f.instance[1::4])           # plane `phase` of every slab
    assert g.grid.voxel_size_zyx_nm == (32.0, 8.0, 8.0)
    assert set(np.unique(g.instance)) <= set(np.unique(f.instance))   # no invented labels


def test_strided_along_another_axis_and_with_an_incomplete_last_slab():
    f = sample("fib25").crop(slice(0, 10), slice(0, 50), slice(0, 30))
    x = f.strided(3, axis=2, phase=1)
    assert x.grid.voxel_size_zyx_nm == (8.0, 8.0, 24.0) and x.grid.shape_zyx == (10, 50, 10)
    assert np.array_equal(x.raw, f.raw[:, :, 1::3]) and np.array_equal(x.instance, f.instance[:, :, 1::3])
    y = f.strided(4, axis=1, mode="slab_mean")
    assert y.grid.voxel_size_zyx_nm == (8.0, 32.0, 8.0) and y.grid.shape_zyx == (10, 12, 30)
    assert np.array_equal(y.raw, np.rint(f.raw[:, :48].reshape(10, 12, 4, 30).mean(axis=2)).astype(np.uint8))
    # 10 planes, step 4: two complete slabs, planes 8 and 9 are dropped for every phase
    for phase in range(4):
        z = f.strided(4, phase=phase)
        assert z.grid.shape_zyx == (2, 50, 30) == z.raw.shape
        assert np.array_equal(z.raw, f.raw[:8][phase::4])
    for bad in (dict(step=0), dict(step=4, phase=4), dict(step=4, axis=3), dict(step=4, mode="mean"), dict(step=11)):
        assert raises(ValueError, f.strided, **bad), bad


# ---- crop --------------------------------------------------------------------------------------
def test_crop_keeps_voxel_size_and_remembers_where_it_was_cut():
    f = sample("fib25")
    c = f.crop(slice(10, 50), slice(None), slice(100, 200))
    assert c.grid.shape_zyx == (40, 384, 100) and c.grid.voxel_size_zyx_nm == f.grid.voxel_size_zyx_nm
    assert np.array_equal(c.raw, f.raw[10:50, :, 100:200])
    assert np.array_equal(c.instance, f.instance[10:50, :, 100:200])
    assert np.array_equal(c.origin_xyz_nm, [100 * 8.0, 0.0, 10 * 8.0])       # (x, y, z) nm
    # ids are kept, tables are recomputed, order is by size inside the crop
    ids = c.instance_ids()
    assert set(ids) == set(np.unique(c.instance)) - {0}
    counts = [c.voxel_count(i) for i in ids]
    assert counts == sorted(counts, reverse=True)
    assert c.voxel_count(ids[0]) == int((c.instance == ids[0]).sum())
    # a crop of a crop adds up
    cc = c.crop(slice(5, 15), slice(20, 30), slice(1, 2))
    assert np.array_equal(cc.raw, f.raw[15:25, 20:30, 101:102])
    assert np.array_equal(cc.origin_xyz_nm, [101 * 8.0, 20 * 8.0, 15 * 8.0])
    # an anisotropic sample: the origin uses (dx, dy, dz) = (6, 6, 30)
    s = sample("snemi3d").crop(slice(2, 4), slice(3, 5), slice(7, 9))
    assert np.array_equal(s.origin_xyz_nm, [7 * 6.0, 3 * 6.0, 2 * 30.0])
    assert raises(ValueError, f.crop, slice(0, 10, 2))            # stride is not a crop
    assert raises(ValueError, f.crop, slice(5, 5))                # nothing selected
    assert raises(TypeError, f.crop, 3)


def test_load_sample_with_crop_reads_the_same_block():
    box = (slice(10, 50), slice(None), slice(100, 200))
    a = S.load_sample("fib25", crop=box)
    b = sample("fib25").crop(*box)
    assert np.array_equal(a.raw, b.raw) and np.array_equal(a.instance, b.instance)
    assert a.grid == b.grid and np.array_equal(a.origin_xyz_nm, b.origin_xyz_nm)
    assert a.derivation == b.derivation
    assert np.array_equal(a.instance_ids(), b.instance_ids())
    assert a.raw.base is None                                     # only the block is held in memory
    assert raises(ValueError, S.load_sample, "fib25", crop=(slice(0, 4),))


def test_every_sample_in_the_manifest_loads_with_its_declared_grid():
    for entry in S.manifest()["samples"]:
        s = sample(entry["name"])
        assert list(s.grid.shape_zyx) == entry["shape_zyx"], entry["name"]
        assert list(s.grid.voxel_size_zyx_nm) == entry["voxel_size_zyx_nm"], entry["name"]
        assert s.has_instances == (entry["label_convention"] != "none"), entry["name"]
        v = s.voxvalue()
        assert v.dtype == np.float32 and 0.0 <= v.min() and v.max() <= 1.0, entry["name"]
        if s.has_instances:
            n = s.attrs["n_instances"]
            assert n == int(s.instance.max()), entry["name"]
            assert np.array_equal(s.instance_ids(), np.arange(1, n + 1)), entry["name"]     # 1 = largest
            assert len(s.skeleton_subjects) == 3, entry["name"]
            assert all(1 <= i <= n for i in s.skeleton_subjects), entry["name"]


def run_as_script(namespace) -> int:
    """Call every test_* function; print PASS / FAIL; return the number of failures."""
    failed = 0
    for name, fn in list(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                traceback.print_exc()
                print("FAIL", name)
    print(f"{failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_as_script(globals()) else 0)
