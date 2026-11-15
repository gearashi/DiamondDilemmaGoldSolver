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
