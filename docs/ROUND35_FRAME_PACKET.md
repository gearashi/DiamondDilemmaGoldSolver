# Four-field classic GPU checkpoints

The classic DFS checkpoint stores 160 rows per lane in four unsigned 16-bit arrays. The paired reader gathers the selected physical cells into four contiguous fields. A single native read returns those arrays directly when the capture fits in one page. WebGPU packs two logical uint16 values in each u32 output word, including pairs that cross an odd row or field boundary. It sizes each page against the storage-binding and buffer limits granted to the device.

## Measurements on the RTX 2060

The prior reader came from commit `9a3403b`. The trial used a frozen Git archive of `2888329`. Both readers used the same 131,072-lane pool and the same field windows. The benchmark warmed both readers, alternated their order across three measured trials and checked SHA-256 equality for all four returned fields. Times are wall-clock medians in milliseconds.

| API | Selected frames | Prior bytes | Paired bytes | Prior maps | Paired maps | Prior ms | Paired ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | One cell | 8 | 8 | 4 | 1 | 0.35 | 0.98 |
| CUDA | Sparse columns | 69,360 | 69,360 | 4 | 1 | 1.60 | 1.06 |
| CUDA | 18 full rows | 18,874,368 | 18,874,368 | 4 | 1 | 12.96 | 13.45 |
| CUDA | Mixed windows | 13,157,664 | 13,157,664 | 4 | 1 | 16.06 | 14.11 |
| CUDA | All 160 rows | 167,772,160 | 167,772,160 | 4 | 1 | 115.06 | 94.21 |
| Vulkan | One cell | 16 | 8 | 4 | 1 | 3.12 | 2.48 |
| Vulkan | Sparse columns | 138,720 | 69,360 | 4 | 1 | 5.58 | 2.31 |
| Vulkan | 18 full rows | 37,748,736 | 18,874,368 | 4 | 1 | 70.30 | 14.75 |
| Vulkan | Mixed windows | 26,315,328 | 13,157,664 | 4 | 1 | 35.10 | 9.88 |
| Vulkan | All 160 rows | 335,544,320 | 167,772,160 | 4 | 2 | 740.41 | 239.06 |
| D3D12 | One cell | 16 | 8 | 4 | 1 | 2.02 | 1.59 |
| D3D12 | Sparse columns | 138,720 | 69,360 | 4 | 1 | 3.70 | 1.62 |
| D3D12 | 18 full rows | 37,748,736 | 18,874,368 | 4 | 1 | 43.33 | 9.54 |
| D3D12 | Mixed windows | 26,315,328 | 13,157,664 | 4 | 1 | 21.60 | 6.93 |
| D3D12 | All 160 rows | 335,544,320 | 167,772,160 | 4 | 2 | 417.36 | 143.69 |

On CUDA the paired gather removes three device-to-host synchronization points from each tested read. One cell and 18 full contiguous rows took longer in this comparison. Both WebGPU backends halved the physical transfer sizes in all five scopes. The ordinary one-window benchmark used the earlier reader's direct range path. The mixed-window comparison used its four-field gather.

## Correctness controls

At the maximum lane count, all 83,886,080 logical values across four fields matched an independent physical read on CUDA, Vulkan and D3D12. The Vulkan and Direct3D tests deliberately granted an 83,886,080-byte maximum storage binding while reading a 167,772,160-byte result. Both required exactly two reads. Additional controls cover nonconsecutive and reordered row windows, noncontiguous lane lists, corrupt high bits in WebGPU's physical words, a changed native cursor, lengths close to 65535, forced small pages and a frozen previous generation. A CPU oracle reconstructs every pair formed across 48 independently shuffled sets of rows.

The initial interleaved packet reduced the same transfer counts and required separate host arrays to unpack the fields. The selected field-major layout returns four contiguous views of one packed result when the capture fits. Multi-page reads assemble the pages in the requested order. The reader preserves the search frontier and its original physical values.

To reproduce a native comparison, extract commit `9a3403b` into a separate directory and run `python -m benchmarks.bench_r35_frame_packet --baseline OLD_TREE --backend Vulkan --width 131072 --output result.json`. The backend also accepts `CUDA` and `D3D12`. The per-field hashes, three paired samples and source identity are in [the measurement record](R35_FRAME_PACKET.json).
