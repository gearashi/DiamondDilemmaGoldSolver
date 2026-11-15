# Classic WebGPU history replay

A resumed classic run already owns a live GPU engine when it asks for native verification of a v1.14 history. The three recorded replica counts were 16, 4 and 8, across six generations and 32 disjoint two-piece jobs from the original puzzle. The controls include formerly legal prefixes that the current Gold mate rules immediately reject. A subprocess imported the actual v1.14.0 engine and job ledger to produce the checkpoint. Replaying that history must disable both later physical-rule tables and construct the historical device buffer accordingly.

The tested implementation reuses the live adapter, logical device and compiled search pipeline when the exact specialized shader and granted buffer limits agree. A historical engine allocates its own search, counter, parameter and policy buffers, bind group and optional counter reducer. Each owner releases its allocations and the last owner destroys the device. For a changed shader, adapter or inadequate limits, the engine builds a separate program. Single-engine searches construct the pipeline once.

| Physical RTX 2060 API | Devices and pipelines, separate → shared | Elapsed seconds, separate → shared |
| --- | --- | --- |
| Vulkan | 4 → 1 | 1.88 → 0.50 |
| Direct3D 12, FXC | 4 → 1 | 135.54 → 30.44 |
| Direct3D 12, Windows SDK DXC | 4 → 1 | 5.68 → 1.91 |

Each path reproduced all 38 native fields, including ownership and paused stacks, from the released CUDA engine. A flipped RNG bit failed verification. A stop callback interrupted replay. Both controls left the live engine available. Closing the original owner before its child and injecting failure during a private buffer upload exercised independence and cleanup. The native build counts are deterministic. Timings include allocation and replay on a shared test host. This comparison measures startup work for historical verification.

[Vulkan](R35_WEBGPU_REPLAY_VULKAN.json) · [Direct3D 12 FXC](R35_WEBGPU_REPLAY_D3D12.json) · [Direct3D 12 DXC](R35_WEBGPU_REPLAY_DXC.json)


## Final decision

Keep device and program reuse for an authenticated bounded historical replay on a compatible adapter. The first replay engine independently rebuilds the candidate buckets from the exact mask bytes. Subsequent engines in that verification can reuse those immutable host arrays. A single historical table buffer can serve consecutive owners on the same program when every input word matches. A concurrent owner or different data gets private storage. After a policy write, the next borrower reuploads the original verified contents. Closure, cancellation and a failed initial upload release resources through the engine and verification scopes.

The original six-generation CUDA history gave the same 38 native fields in all three Windows hardware paths. Compared with separate construction, devices and search pipelines fell from four to one, and CPU lookup builds and device table uploads each fell from four to two. Transfer fell from 935,632 to 497,936 bytes. Two uploads are required here because the live current Gold rules and historical Gold rules differ. The FXC program-only run was slightly faster than the table-sharing run on this host. The reduction in rebuilt and uploaded data was the same across compilers. A different released history visited 512, 4 and 1,024 lanes. A small live device was granted exactly its requested storage limit and needed two extra devices. Reserving the bounded history limit on legacy startup took three devices down to one and retained the requested 64-lane live buffer size. [Vulkan table measurements](R35_REPLAY_TABLE_VULKAN.json) · [FXC](R35_REPLAY_TABLE_FXC.json) · [DXC](R35_REPLAY_TABLE_DXC.json) · [Wide Vulkan](R35_REPLAY_WIDE_VULKAN.json) · [Wide DXC](R35_REPLAY_WIDE_DXC.json)

The single-engine ordinary search already builds once. Keep its normal allocation path. Vulkan, Direct3D 12 FXC and DXC were physically tested. macOS CI attempts the two native lifetime/isolation tests when its runner exposes hardware Metal. Physical Metal remains unverified here. Reopen for a new genuine multi-engine consumer, a different table mutation protocol, a failing adapter test, or a change to the historical replay cap.
