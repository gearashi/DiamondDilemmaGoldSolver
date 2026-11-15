# First and full native GPU stack captures

The classic GPU DFS stores an unsigned 16-bit first index, length, cursor and shift for each of 160 rows in each lane. A new lane starts with the values `(0, 0, 65535, 0)`. In a first capture the reader has no earlier captured values. Full lane reinstallation can overwrite arbitrary tail values, including values beyond the current search depth.

## Physical representation

CUDA and WebGPU measure all four native fields at every row and lane. The first 20 bytes identify the nonempty rows. If all rows are empty, the reader recreates the four arrays immediately. For nonempty rows it reads the measured count, bounds and exact unsigned range of each field. Each row independently selects one of three representations by its size in words.

- A contiguous span includes its default holes in the field ranges.
- A bitmap identifies the nondefault lanes.
- An increasing list identifies widely separated nondefault lanes.

The payload for one record holds the exact four unsigned differences from the measured minima. A constant field takes zero bits. A full page includes a status word checked before decoding. The directory, endpoints, physical default counts, unsigned widths, padding and output ranges are checked. A failed decode discards the prospective fields and makes the normal capture read independent native storage. Optional native decoding shares the source-, platform- and binary-verified physical checkpoint library built by `build_native.py`. A pure NumPy implementation handles bounded packets without a compiler.

## Original and held-out saved work

The frozen original archive has 131,072 lanes and SHA-256 `621d6e7f495977d9b352695c6f6207721619cdbe20f6c1e68b1b291b151a57a1`. It contains 9,223,002 nondefault physical records in 105 rows out of 20,971,520 records. Choosing the most frequent whole tuple per row gave zero additional records to omit on this original. The plain row packet measured 75,041,584 bytes. Field ranges reduced it to 39,453,040 bytes with 15 spans, 78 bitmaps, 12 index lists and 55 empty rows. A full raw logical read is 167,772,160 bytes. A newly allocated 131,072-lane pool returned the same four native fields from 20 bytes.

On the 128-lane saved smoke search there were 6,478 nondefault records out of 20,480. The checked bounded packet costs 25,968 bytes on this input against 163,840 raw bytes. The 480-lane end-of-coverage fixture had 480 nondefaults out of 76,800 and costs 5,044 bytes against 614,400. The original full-width result also survived an ordinary restore over already used native storage. The benchmark compares every native stack field after restoration against the original NPZ field as well as an independently read device snapshot. A direct read costs the same number of bytes if a tail contains an unexpected but valid uint16 bit pattern.

## Decoder and workload controls

Independent CPU encoding reproduced all 83,886,080 original field values. At full width the initial NumPy implementation took a median 1.43 seconds to decode the variable-width packet. The optional checked C implementation took a median 0.133 seconds on the same host in five trials. A separate matched-volume series measured seven exact prefixes of the same packet. The smallest three, containing 63,577, 192,437 and 321,297 nondefault records, took 20, 68 and 107 ms in NumPy and 2, 4 and 5 ms in C under concurrent host work.

CPU negatives include changed directories and counts, fictitious bounds, invalid unsigned widths, truncated or trailing words, nonzero padding, malformed source binding, absent native builds and bias overflow. A full-width constant-frame adverse control reduced the old per-record C decode from 125 ms to 21 ms with contiguous native fills. A zero-payload page uses the checked NumPy span fills, which took 12 ms. Independent finite cases cover all 17 widths from zero to 16 and every starting offset within a 32-bit word. Native tests include empty pools, arbitrary inactive tails, sparse and constant rows, dense uncompressible state, full reinstallation, and exact preservation of an earlier returned generation. The WebGPU jobs exercise the bounded native fixture through software Vulkan and on macOS runners that have an available Metal adapter.


## End-to-end first/full timings

The complete public reader used the same native pool for each arm. Each returned every one of the 18 fields and matched a separate raw device snapshot. Each timing is the median of three alternating runs, in milliseconds, on a 6 GiB GeForce RTX 2060. The direct arm disables the measured default reader and uses the regular CUDA read or the packed WebGPU reader. Forced range coding uses the optional C decoder for nonempty packets. Auto also considers the cost of unpacking a warm pool.

| Backend | State | Direct | Forced range coding | Auto | Auto device reads | Auto measured frame bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CUDA | fresh | 93.1 | 66.7 | 70.1 | 9 | 20 |
| CUDA | installed prefixes | 94.8 | 67.3 | 63.0 | 9 | 20 |
| CUDA | restored | 104.1 | 193.6 | 104.8 | 14 | 4,500 |
| Vulkan | fresh | 218.9 | 84.4 | 98.9 | 3 | 20 |
| Vulkan | installed prefixes | 199.1 | 101.6 | 86.0 | 3 | 20 |
| Vulkan | restored | 259.8 | 285.6 | 295.7 | 5 | 39,453,040 |
| D3D12 | fresh | 347.3 | 144.7 | 131.4 | 3 | 20 |
| D3D12 | installed prefixes | 371.9 | 142.4 | 138.0 | 3 | 20 |
| D3D12 | restored | 433.6 | 580.6 | 427.4 | 5 | 39,453,040 |

Auto device reads cover the entire checkpoint. The measured-frame byte column counts the summary and any bounded-frame packet. A declined packet adds the regular native field transfer to the full checkpoint total. The raw frame size is 167,772,160 bytes on every backend. Original field hashes, full checkpoint transfer totals, per-run timings and earlier plain-row comparisons are in [the measurement record](R35_FRAME_DEFAULTS.json).

## Selected automatic policy

The paired table records commit `6129bea`. On the original warm pool, range coding lost to direct reads on CUDA and Vulkan. The two equivalent C-coded arms on D3D12 were 581 ms and 427 ms against 434 ms direct, so this comparison does not resolve their wall-clock ordering. A forced compact packet saves 128,319,120 transferred frame bytes on this original across all three APIs.

The release uses the already captured native maximum depths and prefix floors to price high-water full reads. An average gap of at least 48 selects an authoritative direct read immediately. The two saved warm pools measured gaps of 69.38 and 49.61 rows. The 480-lane completed control measured zero. This shortcut removes two summary maps and one full native scan of 83,886,080 field values at maximum width. The price hint never supplies a returned frame value. Both CUDA and WebGPU read every returned field from their usual physical storage, including arbitrary unused tails.

For the other pools the reader measures actual nondefaults. Variable records above two per lane on average use the regular full path. Constant rows use their checked measured ranges. The forced benchmark arm preserves the low-transfer option. Reinstallation and independent memory-backed native-interface tests exercise the production selection, including a changed physical tail with identical depth and floor information. The release policy selects the measured direct arm for the original deeply resumed pool on all three APIs. The table gives its measured direct timings.

Run `python build_native.py` to build the optional local decoder. Reproduce an original comparison with `python bench_r35_frame_defaults.py --backend CUDA --tiles PATH_TO_TILES --checkpoint PATH_TO_NPZ --output result.json`. The same command accepts `Vulkan` and `D3D12`.
