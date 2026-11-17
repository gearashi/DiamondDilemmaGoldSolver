# GPU dispatch, Stop and interval checkpoints

The classic GPU solver divides expensive native DFS iterations into pages of physical lanes. Each lane retains its own branch, board and stack. A page boundary gives the search controller time to validate a pending full board, honor Stop, publish the live board or write a due checkpoint. A user who stops partway through a sweep can resume at the next active page. During resizing, complete stacks move with their job IDs, and pending certificates get first priority.

## RTX 2060 measurements

The previous unpaged Direct3D 12 dynamic search at 131,072 lanes triggered a Windows display-driver recovery. The bounded control ran on the original puzzle using forced moves and dynamic MRV. At 512 steps per replica, 1,024 pages took 1,019.29 seconds. Median page duration was 1.004 seconds and the longest page took 1.212 seconds. At 1, 8 and 64 steps the longest pages took 112 ms, 29 ms and 191 ms respectively. At each of the four budgets a separately restored small pool reproduced every compared physical field on the sampled lanes. The system log showed no further driver recovery during this test.

A second test used the public solver with the same maximum replica count, a fixed 512-step budget and a two-second minimum checkpoint interval on CUDA. It published generations 3 and 7 during its first sweep, with the next pages starting at lanes 640 and 2,048. The live board updated during the search. From the Stop request to a verified portable save took 1.282 seconds. At that point 2,580 lanes had tried candidates. These times apply to this machine and checkpoint. The checkpoint interval is measured from the completion of the preceding save. [Hardware measurements](R35_DISPATCH_CONTINUATION.json)

The same public test completed on Direct3D 12 and Vulkan. Direct3D saved at next lanes 256 and 640, then took 2.516 seconds from Stop through its portable checkpoint. Vulkan saved at lanes 384 and 1,024 and took 2.047 seconds. At the maximum 512-step budget, independent all-field comparisons on CUDA and Vulkan also passed all 1,024 pages. Their longest pages took 412 ms and 464 ms respectively.

## Interrupted coverage

In the controlled 256-job comparison, three one-page sessions on the predecessor each advanced the same first 128 jobs by one more DFS step. With continuation, two such sessions advanced all 256 distinct jobs once. A second control captured an interval checkpoint and refreshed the live board partway through a 512-job sweep. All 512 jobs advanced once. The initial branch order and full uninterrupted search have the same physical endpoints. Actual CUDA, Vulkan and Direct3D tests checked every physical field after consecutive partial stops and cold restoration.

The tests also cover an older checkpoint with no scheduling hint, a smaller restored pool, malformed optional scheduling data, a multi-loop candidate before the genuine Gold on the same page, loss of the separate solution files, and failed publication while restoring from a checkpoint. [Public runner tests](../test_r35_public_pages.py) · [Native dispatch tests](../test_r35_dispatch_pages.py)

Run `python -m benchmarks.bench_r35_gpu_pages --help` for direct physical replay and `bench_r35_public_checkpoints.py --help` for the isolated public dashboard, checkpoint and Stop test. Both preserve independently validated Gold output.
