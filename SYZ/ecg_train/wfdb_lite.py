"""wfdb_lite -- minimal WFDB reader in pure numpy.

Reads PhysioNet-style records: a ``.hea`` text header plus signal samples held
either in a ``.dat`` binary or in a MATLAB v5 ``.mat`` file (the layout used by
the PhysioNet/CinC Challenge sets, where the ``.hea`` still carries gain and
baseline and the ``.mat`` only holds the raw integer matrix).

Only numpy and the standard library are used, so this file can ship inside the
inference package without dragging scipy along.

Supported ``.dat`` formats: 8, 16, 24, 32, 61, 80, 160, 212.
Supported ``.mat`` files: MATLAB level-5 (including zlib-compressed elements).
MATLAB v7.3 files are HDF5 and are rejected with an explicit message.
"""

from __future__ import annotations

import os
import re
import struct
import zlib

import numpy as np

# Canonical 12-lead order every downstream stage assumes.
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6")

# Spellings seen in the wild -> canonical name.
_LEAD_ALIASES = {
    "i": "I", "lead i": "I", "mli": "I",
    "ii": "II", "lead ii": "II", "mlii": "II",
    "iii": "III", "lead iii": "III",
    "avr": "aVR", "-avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
}


class WFDBError(Exception):
    """Raised when a record cannot be parsed or its signal file is unusable."""


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------

_GAIN_RE = re.compile(r"^(?P<gain>[-+0-9.eE]+)"
                      r"(?:\((?P<baseline>[-+0-9]+)\))?"
                      r"(?:/(?P<units>\S+))?$")


class Header:
    """Parsed contents of a ``.hea`` file."""

    def __init__(self, name, n_sig, fs, n_samp, signals, comments):
        self.name = name
        self.n_sig = n_sig
        self.fs = fs
        self.n_samp = n_samp
        self.signals = signals      # list of per-signal dicts
        self.comments = comments    # list of '#' lines, verbatim

    @property
    def lead_names(self):
        return [s["desc"] for s in self.signals]


def _parse_signal_line(line):
    """Parse one signal specification line of a WFDB header."""
    parts = line.split()
    if len(parts) < 2:
        raise WFDBError("signal line has too few fields: %r" % line)

    spec = {"filename": parts[0]}

    # Format field: "16", "16x2", "212+24", "16:3" ... only the leading integer
    # is the storage format; the rest are frame/skew/offset modifiers.
    fmt_field = parts[1]
    m = re.match(r"^(\d+)", fmt_field)
    if not m:
        raise WFDBError("unreadable format field: %r" % fmt_field)
    spec["fmt"] = int(m.group(1))

    m = re.search(r"x(\d+)", fmt_field)
    spec["samples_per_frame"] = int(m.group(1)) if m else 1
    m = re.search(r"\+(\d+)", fmt_field)
    spec["byte_offset"] = int(m.group(1)) if m else 0

    gain, baseline, units = 200.0, 0, "mV"
    if len(parts) >= 3:
        m = _GAIN_RE.match(parts[2])
        if m:
            gain = float(m.group("gain"))
            if m.group("baseline") is not None:
                baseline = int(m.group("baseline"))
            if m.group("units"):
                units = m.group("units")
    # A gain of 0 means "uncalibrated" in the WFDB spec; treat as unity.
    if gain == 0:
        gain = 200.0
    spec["gain"] = gain
    spec["units"] = units

    spec["adc_res"] = int(parts[3]) if len(parts) >= 4 and parts[3].lstrip("-").isdigit() else 0
    spec["adc_zero"] = int(parts[4]) if len(parts) >= 5 and parts[4].lstrip("-").isdigit() else 0

    # Field 3 carries the baseline only when it was absent from the gain field.
    if len(parts) >= 3 and "(" not in parts[2]:
        baseline = spec["adc_zero"]
    spec["baseline"] = baseline

    spec["desc"] = " ".join(parts[8:]).strip() if len(parts) >= 9 else ""
    return spec


def read_header(hea_path):
    """Read and parse a ``.hea`` file."""
    with open(hea_path, "r", errors="replace") as fh:
        raw = fh.read().splitlines()

    comments = [ln for ln in raw if ln.startswith("#")]
    lines = [ln.strip() for ln in raw
             if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise WFDBError("empty header: %s" % hea_path)

    fields = lines[0].split()
    if len(fields) < 2:
        raise WFDBError("bad record line: %r" % lines[0])

    name = fields[0].split("/")[0]
    n_sig = int(fields[1])
    fs = float(fields[2]) if len(fields) >= 3 else 250.0
    n_samp = int(fields[3]) if len(fields) >= 4 else 0

    signals = [_parse_signal_line(ln) for ln in lines[1:1 + n_sig]]
    if len(signals) != n_sig:
        raise WFDBError("header declares %d signals but lists %d"
                        % (n_sig, len(signals)))
    return Header(name, n_sig, fs, n_samp, signals, comments)


# --------------------------------------------------------------------------
# .dat decoding
# --------------------------------------------------------------------------

def _decode_212(buf, n_values):
    """Format 212: two 12-bit signed samples packed into three bytes."""
    n_triplets = (n_values + 1) // 2
    need = n_triplets * 3
    if buf.size < need:
        raise WFDBError("format 212 stream is short: have %d bytes, need %d"
                        % (buf.size, need))
    b = buf[:need].reshape(-1, 3).astype(np.int32)

    first = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    second = b[:, 2] | ((b[:, 1] >> 4) << 8)

    # 12-bit two's complement -> signed
    first = np.where(first > 2047, first - 4096, first)
    second = np.where(second > 2047, second - 4096, second)

    out = np.empty(n_triplets * 2, dtype=np.int32)
    out[0::2] = first
    out[1::2] = second
    return out[:n_values]


def _decode_24(buf, n_values):
    need = n_values * 3
    if buf.size < need:
        raise WFDBError("format 24 stream is short")
    b = buf[:need].reshape(-1, 3).astype(np.int32)
    val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    return np.where(val > 0x7FFFFF, val - 0x1000000, val)


def _decode_dat(path, fmt, n_sig, n_samp, byte_offset):
    """Read a signal file and return an int32 array shaped (n_sig, n_samp)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if byte_offset:
        raw = raw[byte_offset:]

    if fmt == 16:
        flat = raw.view(np.int16).astype(np.int32) if raw.size % 2 == 0 \
            else raw[:raw.size - raw.size % 2].view(np.int16).astype(np.int32)
    elif fmt == 61:
        usable = raw[:raw.size - raw.size % 2]
        flat = usable.view(">i2").astype(np.int32)
    elif fmt == 32:
        usable = raw[:raw.size - raw.size % 4]
        flat = usable.view(np.int32).astype(np.int32)
    elif fmt == 160:
        usable = raw[:raw.size - raw.size % 2]
        flat = usable.view(np.uint16).astype(np.int32) - 32768
    elif fmt == 80:
        flat = raw.astype(np.int32) - 128
    elif fmt == 8:
        # Format 8 stores first differences, not absolute samples.
        flat = np.cumsum(raw.view(np.int8).astype(np.int32))
    elif fmt == 212:
        flat = _decode_212(raw, n_sig * n_samp if n_samp else raw.size * 2 // 3)
    elif fmt == 24:
        flat = _decode_24(raw, n_sig * n_samp if n_samp else raw.size // 3)
    else:
        raise WFDBError("unsupported WFDB format %d in %s" % (fmt, path))

    if n_samp:
        need = n_sig * n_samp
        if flat.size < need:
            raise WFDBError("%s holds %d samples, header declares %d"
                            % (path, flat.size, need))
        flat = flat[:need]
    else:
        flat = flat[:flat.size - flat.size % n_sig]

    # WFDB interleaves signals sample by sample.
    return flat.reshape(-1, n_sig).T


# --------------------------------------------------------------------------
# MATLAB level-5 reader
# --------------------------------------------------------------------------

_MI_TYPES = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 9: np.float64,
    12: np.int64, 13: np.uint64, 16: np.uint8, 17: np.uint8, 18: np.uint8,
}


def _read_tag(buf, pos, endian):
    """Read an element tag, handling the compact small-data-element form."""
    (word,) = struct.unpack_from(endian + "I", buf, pos)
    small = (word >> 16) & 0xFFFF
    if small:                                   # small data element
        return word & 0xFFFF, small, pos + 4, True
    (nbytes,) = struct.unpack_from(endian + "I", buf, pos + 4)
    return word, nbytes, pos + 8, False


def _element_end(pos, nbytes, data_pos, is_small):
    """Byte offset just past an element; non-small elements pad to 8 bytes."""
    if is_small:
        return pos + 8
    return data_pos + nbytes + (-nbytes) % 8


def _read_element(buf, pos, endian):
    """Return (dtype_code, payload, next_pos) for one MAT element.

    ``payload`` is a numpy array for numeric types, and raw ``bytes`` for the
    two container types: miCOMPRESSED (15, already inflated) and miMATRIX (14).
    """
    dtype_code, nbytes, data_pos, is_small = _read_tag(buf, pos, endian)
    end = _element_end(pos, nbytes, data_pos, is_small)

    if dtype_code == 15:                        # miCOMPRESSED
        return 15, zlib.decompress(bytes(buf[data_pos:data_pos + nbytes])), end
    if dtype_code == 14:                        # miMATRIX
        return 14, bytes(buf[data_pos:data_pos + nbytes]), end

    np_dtype = _MI_TYPES.get(dtype_code)
    if np_dtype is None:
        raise WFDBError("unsupported MAT data type %d" % dtype_code)

    itemsize = np.dtype(np_dtype).itemsize
    arr = np.frombuffer(bytes(buf[data_pos:data_pos + nbytes]),
                        dtype=np.dtype(endian + np.dtype(np_dtype).str[1:]),
                        count=nbytes // itemsize)
    return dtype_code, arr, end


def _parse_matrix(buf, endian):
    """Parse a miMATRIX payload into (name, ndarray). Numeric arrays only."""
    pos = 0
    _, flags, pos = _read_element(buf, pos, endian)
    mclass = int(flags[0]) & 0xFF
    is_complex = bool((int(flags[0]) >> 8) & 0x08)

    _, dims, pos = _read_element(buf, pos, endian)
    _, name_arr, pos = _read_element(buf, pos, endian)
    name = bytes(np.asarray(name_arr, dtype=np.uint8)).decode("latin-1").strip("\x00")

    if mclass in (1, 2, 3, 4, 5):               # cell/struct/object/char
        return name, None
    if is_complex:
        return name, None

    _, real, pos = _read_element(buf, pos, endian)
    dims = np.asarray(dims, dtype=np.int64)
    # MATLAB stores column-major.
    arr = np.asarray(real).reshape(tuple(dims[::-1]))[...].T \
        if dims.size == 2 else np.asarray(real)
    return name, np.ascontiguousarray(arr)


def read_mat_v5(path):
    """Read a MATLAB level-5 file into ``{name: ndarray}``."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if len(buf) < 128:
        raise WFDBError("%s is too small to be a MAT file" % path)

    if buf[:8] == b"\x89HDF\r\n\x1a\n":
        raise WFDBError(
            "%s is a MATLAB v7.3 (HDF5) file; re-save it as '-v7' or use the "
            ".dat form of this record" % path)

    endian = "<" if buf[126:128] == b"IM" else ">"

    out = {}

    def walk(block, pos, depth=0):
        """Consume MAT elements from ``block``, following compressed ones."""
        if depth > 4:
            return
        n = len(block)
        while pos + 8 <= n:
            try:
                code, payload, pos = _read_element(block, pos, endian)
            except Exception:
                return
            if code == 15:                      # compressed -> nested elements
                walk(payload, 0, depth + 1)
            elif code == 14:                    # miMATRIX
                name, arr = _parse_matrix(payload, endian)
                if arr is not None:
                    out[name] = arr
            # anything else at this level is padding or unsupported; skip it

    walk(buf, 128)
    return out


def _load_mat_signal(path, n_sig, n_samp):
    """Pull the sample matrix out of a Challenge-style ``.mat`` file."""
    try:
        variables = read_mat_v5(path)
    except WFDBError:
        raise
    except Exception as exc:                    # malformed / exotic encoding
        try:                                    # scipy is optional, dev-only
            from scipy.io import loadmat        # noqa: PLC0415
            variables = {k: v for k, v in loadmat(path).items()
                         if not k.startswith("__")}
        except Exception:
            raise WFDBError("cannot read %s: %s" % (path, exc))

    if not variables:
        raise WFDBError("%s contains no readable variables" % path)

    arr = variables.get("val")
    if arr is None:
        # Fall back to the largest 2-D numeric variable present.
        arr = max(variables.values(), key=lambda a: np.asarray(a).size)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] != n_sig and arr.shape[1] == n_sig:
        arr = arr.T
    if arr.shape[0] != n_sig:
        raise WFDBError("%s holds shape %s, header declares %d signals"
                        % (path, arr.shape, n_sig))
    if n_samp and arr.shape[1] < n_samp:
        raise WFDBError("%s holds %d samples, header declares %d"
                        % (path, arr.shape[1], n_samp))
    return arr[:, :n_samp] if n_samp else arr


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def canonical_lead(desc):
    """Map a header lead description onto a standard 12-lead name, or None."""
    key = str(desc).strip().lower().replace("lead", "").strip()
    key = key.replace(" ", "").replace("-", "").replace("_", "")
    if key in _LEAD_ALIASES:
        return _LEAD_ALIASES[key]
    return _LEAD_ALIASES.get(str(desc).strip().lower())


def read_record(path, leads=STANDARD_LEADS):
    """Read a record and return ``(signal, fs, lead_names)``.

    ``path`` may point at the ``.hea``, the ``.dat``/``.mat``, or carry no
    extension at all. ``signal`` is float32 in millivolts, shaped
    ``(len(leads), n_samples)`` when ``leads`` is given -- missing leads come
    back as zeros -- otherwise ``(n_sig, n_samples)`` in header order.
    """
    base = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    hea_path = base + ".hea"
    if not os.path.exists(hea_path):
        raise WFDBError("header not found: %s" % hea_path)

    hdr = read_header(hea_path)
    directory = os.path.dirname(os.path.abspath(hea_path))

    # Group signals by the file they live in; nearly always a single file.
    raw = np.zeros((hdr.n_sig, hdr.n_samp), dtype=np.int32) if hdr.n_samp else None
    by_file = {}
    for idx, spec in enumerate(hdr.signals):
        by_file.setdefault(spec["filename"], []).append(idx)

    for fname, idxs in by_file.items():
        sig_path = os.path.join(directory, fname)
        if not os.path.exists(sig_path):
            alt = os.path.join(directory, os.path.basename(base) + ".mat")
            if os.path.exists(alt):
                sig_path = alt
            else:
                raise WFDBError("signal file not found: %s" % sig_path)

        fmt = hdr.signals[idxs[0]]["fmt"]
        if sig_path.lower().endswith(".mat"):
            block = _load_mat_signal(sig_path, len(idxs), hdr.n_samp)
        else:
            block = _decode_dat(sig_path, fmt, len(idxs), hdr.n_samp,
                                hdr.signals[idxs[0]]["byte_offset"])
        if raw is None:
            raw = np.zeros((hdr.n_sig, block.shape[1]), dtype=np.int32)
        n = min(raw.shape[1], block.shape[1])
        for slot, sig_idx in enumerate(idxs):
            raw[sig_idx, :n] = block[slot, :n]

    if raw is None:
        raise WFDBError("no samples decoded for %s" % base)

    phys = np.empty(raw.shape, dtype=np.float32)
    for i, spec in enumerate(hdr.signals):
        phys[i] = (raw[i].astype(np.float64) - spec["baseline"]) / spec["gain"]

    # WFDB marks missing samples with the most negative storage value; those
    # become absurd physical values, so clamp them to zero.
    phys[~np.isfinite(phys)] = 0.0

    if leads is None:
        return phys, hdr.fs, hdr.lead_names

    present = {}
    for i, desc in enumerate(hdr.lead_names):
        name = canonical_lead(desc)
        if name is not None and name not in present:
            present[name] = i

    out = np.zeros((len(leads), phys.shape[1]), dtype=np.float32)
    missing = []
    for row, name in enumerate(leads):
        if name in present:
            out[row] = phys[present[name]]
        else:
            missing.append(name)

    if len(missing) == len(leads):
        # Header carried no recognisable lead names -- fall back to file order.
        k = min(len(leads), phys.shape[0])
        out[:k] = phys[:k]
        missing = list(leads[k:])

    return out, hdr.fs, list(leads)


def record_length_seconds(path):
    """Duration of a record in seconds, read from the header alone."""
    base = os.path.splitext(path)[0] if os.path.splitext(path)[1] else path
    hdr = read_header(base + ".hea")
    return hdr.n_samp / hdr.fs if hdr.fs else 0.0
