"""figure -- captions, panel letters, montages and scale bars for pictures.  CPU only: Pillow + numpy.

Every function takes a picture in any of these forms

    * an object with an ``rgba`` attribute (an ``emray.Image``),
    * a numpy array (H, W) grey, (H, W, 3) RGB or (H, W, 4) RGBA; uint8, bool, or float in [0, 1],
    * a ``PIL.Image.Image``,

and returns a new ``PIL.Image.Image`` (which has ``.save(path)`` and shows inline in Jupyter).
Sizes, paddings and positions are in pixels.  Text is drawn with Pillow's built-in scalable font,
so no font file is referenced; lengths are labelled in plain ASCII ("um" for micrometre), because
that font is not certain to contain a micro sign.

    panels = [scale_bar(img, cam.pixel_size()) for img in images]
    show(montage(panels, captions=["index space", "physical space"], labels=True))
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

__all__ = ["to_pil", "to_array", "to_png_bytes", "show", "caption", "label_panel", "letters", "montage",
           "side_by_side", "scale_bar", "round_length", "format_length"]

WHITE, BLACK = (255, 255, 255), (0, 0, 0)


# --------------------------------------------------------------------------------------------
# conversions
# --------------------------------------------------------------------------------------------
def to_pil(image):
    """Any supported picture -> ``PIL.Image`` in mode "RGB" or "RGBA" (a copy; the input is untouched)."""
    if isinstance(image, PILImage.Image):
        return image.copy() if image.mode in ("RGB", "RGBA") else image.convert("RGBA" if "A" in image.mode else "RGB")
    a = np.asarray(image.rgba if hasattr(image, "rgba") else image)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4) or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"a picture must have shape (H, W), (H, W, 3) or (H, W, 4), got {a.shape}")
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or a.min() < 0.0 or a.max() > 1.0:
            raise ValueError("a float picture must hold finite values in [0, 1]")
        a = np.floor(a * 255.0 + 0.5).astype(np.uint8)
    elif a.dtype != np.uint8:
        raise TypeError(f"a picture must be uint8, bool or float in [0, 1], got {a.dtype}")
    return PILImage.fromarray(np.ascontiguousarray(a))      # (H, W, 3) -> "RGB", (H, W, 4) -> "RGBA"


def to_array(image):
    """Any supported picture -> uint8 array (H, W, 3) or (H, W, 4)."""
    return np.asarray(to_pil(image)).copy()


def to_png_bytes(image):
    """Any supported picture -> the bytes of a PNG file."""
    buffer = io.BytesIO()
    to_pil(image).save(buffer, format="PNG")
    return buffer.getvalue()


def show(image):
    """Display a picture.  Inside a Jupyter kernel it is shown inline (``IPython.display``) and
    nothing is returned, so a cell ending in ``show(x)`` does not show it twice.  Anywhere else the
    ``PIL.Image`` is returned (call ``.save(path)`` or ``.show()`` on it)."""
    picture = to_pil(image)
    try:
        import IPython
        from IPython import display as ipython_display
    except ImportError:
        return picture
    shell = IPython.get_ipython()
    if shell is None or getattr(shell, "kernel", None) is None:
        return picture
    ipython_display.display(picture)
    return None


# --------------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------------
def _font(size):
    """Pillow's built-in scalable font at ``size`` pixels (no font file path is involved)."""
    return ImageFont.load_default(size=float(size))


def _default_size(width):
    """Text size that reads well on a picture ``width`` pixels wide: 1/32 of it, at least 12."""
    return max(12, int(width) // 32)


def _wrap(text, font, max_width):
    """Split ``text`` into lines no wider than ``max_width`` pixels (explicit newlines are kept;
    a single word that is too long stays on its own line)."""
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            trial = word if not line else line + " " + word
            if line and font.getlength(trial) > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _line_height(font):
    ascent, descent = font.getmetrics()
    return ascent + descent


def caption(image, text, size=None, position="top", color=BLACK, background=WHITE, pad=None, lines=None):
    """Add a strip with ``text`` above (``position="top"``) or below (``"bottom"``) the picture.
    The picture itself is not touched: the result is taller by the height of the strip.  Text is
    wrapped to the width of the picture; ``size`` is the text height in pixels (default: 1/32 of
    the width, at least 12); ``lines`` reserves room for at least that many lines, so that
    captions of different length give strips of equal height."""
    if position not in ("top", "bottom"):
        raise ValueError(f'position must be "top" or "bottom", got {position!r}')
    picture = to_pil(image)
    size = _default_size(picture.width) if size is None else size
    pad = max(2, size // 3) if pad is None else pad
    font = _font(size)
    wrapped = _wrap(text, font, picture.width - 2 * pad)
    step = _line_height(font) + max(1, size // 6)
    strip = int(math.ceil(max(len(wrapped), lines or 0) * step + 2 * pad))      # (a text size may be a fraction)
    out = PILImage.new(picture.mode, (picture.width, picture.height + strip), _fill(background, picture.mode))
    out.paste(picture, (0, strip if position == "top" else 0))
    draw = ImageDraw.Draw(out)
    top = 0 if position == "top" else picture.height
    for n, line in enumerate(wrapped):
        draw.text((pad, top + pad + n * step), line, fill=_fill(color, picture.mode), font=font)
    return out


def letters(n):
    """Panel letters ``["a", "b", ...]`` (after "z": "aa", "ab", ...)."""
    out = []
    for i in range(int(n)):
        name = ""
        i += 1
        while i > 0:
            i, r = divmod(i - 1, 26)
            name = chr(ord("a") + r) + name
        out.append(name)
    return out


def label_panel(image, letter, size=None, corner="top-left", color=BLACK, background=WHITE, pad=None):
    """Draw a panel letter ("a", "b", ...) in a small filled box in one corner of the picture
    (``corner`` is "top-left", "top-right", "bottom-left" or "bottom-right").  Unlike
    :func:`caption` this covers the pixels under the box.  ``size`` defaults to 1/16 of the
    width, at least 14."""
    if corner not in ("top-left", "top-right", "bottom-left", "bottom-right"):
        raise ValueError(f"unknown corner {corner!r}")
    picture = to_pil(image)
    size = max(14, picture.width // 16) if size is None else size
    pad = max(2, size // 4) if pad is None else pad
    font = _font(size)
    box_w = int(math.ceil(font.getlength(str(letter)))) + 2 * pad
    box_h = _line_height(font) + 2 * pad
    x0 = 0 if "left" in corner else picture.width - box_w
    y0 = 0 if "top" in corner else picture.height - box_h
    draw = ImageDraw.Draw(picture)
    draw.rectangle([x0, y0, x0 + box_w - 1, y0 + box_h - 1], fill=_fill(background, picture.mode))
    draw.text((x0 + pad, y0 + pad), str(letter), fill=_fill(color, picture.mode), font=font)
    return picture


def _fill(color, mode):
    """An RGB(A) colour (0..255) as a tuple that fits the PIL ``mode``."""
    c = tuple(int(v) for v in color)
    if len(c) not in (3, 4):
        raise ValueError(f"a colour must be (r, g, b) or (r, g, b, a) in 0..255, got {color!r}")
    return (c + (255,))[:4] if mode == "RGBA" else c[:3]


# --------------------------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------------------------
def montage(images, captions=None, ncols=None, pad=8, background=WHITE, labels=None, size=None, color=None):
    """Arrange pictures in a grid, left to right and top to bottom.

    ``captions``  one text per picture, drawn in a strip above it (strips get equal height);
    ``labels``    ``True`` for panel letters a, b, c, ... or a list of strings.  With captions the
                  letter goes in front of the caption, "(a) ...", and no pixel of the picture is
                  covered; without captions it is drawn on the picture (see :func:`label_panel`);
    ``ncols``     pictures per row (default: all in one row up to 4, else the smallest square grid);
    ``pad``       pixels of ``background`` between the cells and around the grid;
    ``size``      text height of the captions in pixels;
    ``color``     colour of the caption text (default: black on a bright ``background``, white on
                  a dark one, so that captions never vanish).

    Pictures are never rescaled.  Every cell has the size of the largest picture; smaller ones
    are centred horizontally and aligned to the top of their cell."""
    pictures = [to_pil(im) for im in images]
    n = len(pictures)
    if n == 0:
        raise ValueError("montage needs at least one picture")
    if labels is True:
        labels = letters(n)
    for name, texts in (("captions", captions), ("labels", labels)):
        if texts is not None and len(texts) != n:
            raise ValueError(f"{len(texts)} {name} for {n} pictures")
    if labels is not None and captions is not None:
        captions = [f"({letter}) {text}" for letter, text in zip(labels, captions)]
    elif labels is not None:
        pictures = [label_panel(p, t) for p, t in zip(pictures, labels)]
    if captions is not None:
        size = _default_size(max(p.width for p in pictures)) if size is None else size
        pad_text = max(2, size // 3)
        most = max(len(_wrap(t, _font(size), p.width - 2 * pad_text)) for p, t in zip(pictures, captions))
        if color is None:                               # the text must stand out from the strip
            color = BLACK if sum(_fill(background, "RGB")) >= 384 else WHITE
        pictures = [caption(p, t, size=size, pad=pad_text, color=color, background=background, lines=most)
                    for p, t in zip(pictures, captions)]
    if ncols is None:
        ncols = n if n <= 4 else int(math.ceil(math.sqrt(n)))
    ncols = max(1, min(int(ncols), n))
    nrows = int(math.ceil(n / ncols))
    cell_w, cell_h = max(p.width for p in pictures), max(p.height for p in pictures)
    mode = "RGBA" if any(p.mode == "RGBA" for p in pictures) else "RGB"
    out = PILImage.new(mode, (pad + ncols * (cell_w + pad), pad + nrows * (cell_h + pad)), _fill(background, mode))
    for i, p in enumerate(pictures):
        row, col = divmod(i, ncols)
        out.paste(p, (pad + col * (cell_w + pad) + (cell_w - p.width) // 2, pad + row * (cell_h + pad)))
    return out


def side_by_side(*images, captions=None, **options):
    """:func:`montage` with all pictures in one row: ``side_by_side(index, physical, captions=[...])``."""
    if len(images) == 1 and isinstance(images[0], (list, tuple)):
        images = tuple(images[0])
    return montage(list(images), captions=captions, ncols=max(1, len(images)), **options)


# --------------------------------------------------------------------------------------------
# scale bar
# --------------------------------------------------------------------------------------------
def round_length(max_length):
    """The largest "round" length not above ``max_length``: 1, 2 or 5 times a power of ten
    (0.35 -> 0.2, 3.07 -> 2, 700 -> 500)."""
    if not (np.isfinite(max_length) and max_length > 0):
        raise ValueError(f"max_length must be positive, got {max_length!r}")
    power = 10.0 ** math.floor(math.log10(max_length))
    for mantissa in (5.0, 2.0, 1.0):
        if mantissa * power <= max_length * (1 + 1e-12):
            return mantissa * power
    return power


def format_length(length_um):
    """A length in micrometres as text with a sensible unit: 0.05 -> "50 nm", 2 -> "2 um",
    1500 -> "1.5 mm"."""
    if length_um < 1.0:
        return f"{length_um * 1000.0:.4g} nm"
    if length_um < 1000.0:
        return f"{length_um:.4g} um"
    return f"{length_um / 1000.0:.4g} mm"


def scale_bar(image, pixel_size, length=None, label=None, position="bottom-right", color=WHITE, outline=BLACK,
              thickness=None, margin=None, size=None):
    """Overlay a scale bar on an ORTHOGRAPHIC picture (a perspective picture has no single scale).
    Use it on pictures of PHYSICAL space only: in index space the voxels are drawn as cubes, so a
    length on the picture is not a physical length (that is the mistake index space shows).

    ``pixel_size``  edge of one pixel in scene units = micrometres (``Camera.pixel_size()``);
    ``length``      of the bar in micrometres; default: the largest round length (1, 2 or 5 x 10^n)
                    that is at most a quarter of the picture width;
    ``label``       text above the bar; default ``format_length(length)``, ``""`` for none;
    ``position``    "bottom-right", "bottom-left", "top-right" or "top-left".

    The bar is ``round(length / pixel_size)`` pixels long, drawn in ``color`` with a one-pixel
    ``outline`` around it so that it reads on dark and on bright pictures."""
    if position not in ("bottom-right", "bottom-left", "top-right", "top-left"):
        raise ValueError(f"unknown position {position!r}")
    if not (np.isfinite(pixel_size) and pixel_size > 0):
        raise ValueError(f"pixel_size must be positive, got {pixel_size!r}")
    picture = to_pil(image)
    if length is None:
        length = round_length(0.25 * picture.width * pixel_size)
    bar = int(round(length / pixel_size))
    size = _default_size(picture.width) if size is None else size
    thickness = max(2, size // 4) if thickness is None else thickness
    margin = max(4, size) if margin is None else margin
    if bar < 1 or bar + 2 * margin > picture.width:
        raise ValueError(f"a bar of {length:g} um is {bar} px long and does not fit a picture {picture.width} px wide")
    text = format_length(length) if label is None else str(label)
    font = _font(size)
    x0 = picture.width - margin - bar if "right" in position else margin
    y0 = picture.height - margin - thickness if "bottom" in position else margin + _line_height(font) + 4
    draw = ImageDraw.Draw(picture)
    draw.rectangle([x0 - 1, y0 - 1, x0 + bar, y0 + thickness], fill=_fill(outline, picture.mode))
    draw.rectangle([x0, y0, x0 + bar - 1, y0 + thickness - 1], fill=_fill(color, picture.mode))
    if text:
        tx = x0 + (bar - font.getlength(text)) / 2
        tx = min(max(tx, 2), picture.width - font.getlength(text) - 2)
        draw.text((tx, y0 - _line_height(font) - 4), text, fill=_fill(color, picture.mode), font=font,
                  stroke_width=max(1, size // 12), stroke_fill=_fill(outline, picture.mode))
    return picture
