# Benchmark drivers

Standalone benchmark drivers live here to keep the repository root browsable. Shared benchmark helpers and drivers imported by tests remain at the root.

Run these drivers as Python modules **from the repository root**, using the environment installed by the setup scripts:

```powershell
.\venv\Scripts\python.exe -B -m benchmarks.bench_r35_strand_matching --help
```

On Linux or macOS:

```sh
./venv/bin/python -B -m benchmarks.bench_r35_strand_matching --help
```

For any moved `NAME.py`, replace the old `python NAME.py ...` command with `python -m benchmarks.NAME ...`. Do not run `python benchmarks/NAME.py`: module execution keeps imports of solver code and shared helpers working. Existing arguments, input paths and output paths still apply relative to the repository root.

Benchmarks are research/reproduction tools, not the dashboard launcher. Some need optional solver or GPU dependencies, prepared puzzle data, or locally saved checkpoints. Read each driver's arguments and the linked round reports before running it. Choose an isolated output directory for new results.

## Drivers by round

### Round 24

- [benchmark_hybrid_round24.py](benchmark_hybrid_round24.py)
- [benchmark_learning_round24.py](benchmark_learning_round24.py)
- [benchmark_regions_round24.py](benchmark_regions_round24.py)

### Round 27

- [bench_r27_incumbent_followups.py](bench_r27_incumbent_followups.py)
- [bench_r27_incumbent_perturb.py](bench_r27_incumbent_perturb.py)
- [bench_r27_incumbent_rewiring.py](bench_r27_incumbent_rewiring.py)
- [bench_r27_incumbent_trades.py](bench_r27_incumbent_trades.py)
- [bench_r27_structure.py](bench_r27_structure.py)
- [benchmark_r27_cuda_original.py](benchmark_r27_cuda_original.py)

### Round 28

- [bench_r28_device.py](bench_r28_device.py)
- [bench_r28_exchange.py](bench_r28_exchange.py)
- [bench_r28_geo.py](bench_r28_geo.py)
- [bench_r28_logic.py](bench_r28_logic.py)

### Round 29

- [bench_r29_exchange.py](bench_r29_exchange.py)
- [bench_r29_gold.py](bench_r29_gold.py)
- [bench_r29_plateau.py](bench_r29_plateau.py)
- [bench_r29_regional.py](bench_r29_regional.py)

### Round 30

- [bench_r30_routes.py](bench_r30_routes.py)

### Round 31

- [bench_r31_fresh.py](bench_r31_fresh.py)
- [bench_r31_fresh_followups.py](bench_r31_fresh_followups.py)
- [bench_r31_gold.py](bench_r31_gold.py)

### Round 32

- [bench_r32_fresh.py](bench_r32_fresh.py)
- [bench_r32_gold.py](bench_r32_gold.py)
- [bench_r32_gold_followups.py](bench_r32_gold_followups.py)
- [bench_r32_gpu.py](bench_r32_gpu.py)
- [bench_r32_route_followups.py](bench_r32_route_followups.py)
- [bench_r32_route_settings.py](bench_r32_route_settings.py)

### Round 33

- [bench_r33_gold_buckets.py](bench_r33_gold_buckets.py)
- [bench_r33_gold_lookup.py](bench_r33_gold_lookup.py)

### Round 34

- [bench_r34_band_placements.py](bench_r34_band_placements.py)
- [bench_r34_physical.py](bench_r34_physical.py)
- [bench_r34_physical_followups.py](bench_r34_physical_followups.py)
- [bench_r34_repair.py](bench_r34_repair.py)

### Round 35

- [bench_r35_alternative_retention.py](bench_r35_alternative_retention.py)
- [bench_r35_assembly_relocation.py](bench_r35_assembly_relocation.py)
- [bench_r35_blank_projection.py](bench_r35_blank_projection.py)
- [bench_r35_board_codec.py](bench_r35_board_codec.py)
- [bench_r35_column_copy_cost.py](bench_r35_column_copy_cost.py)
- [bench_r35_comb_pricing.py](bench_r35_comb_pricing.py)
- [bench_r35_complete_weak.py](bench_r35_complete_weak.py)
- [bench_r35_contact_assignment.py](bench_r35_contact_assignment.py)
- [bench_r35_contact_bound.py](bench_r35_contact_bound.py)
- [bench_r35_contact_counterexamples.py](bench_r35_contact_counterexamples.py)
- [bench_r35_contact_dual_portfolios.py](bench_r35_contact_dual_portfolios.py)
- [bench_r35_contact_exact_proof.py](bench_r35_contact_exact_proof.py)
- [bench_r35_contact_global.py](bench_r35_contact_global.py)
- [bench_r35_contact_global_witness.py](bench_r35_contact_global_witness.py)
- [bench_r35_contact_integer_opt.py](bench_r35_contact_integer_opt.py)
- [bench_r35_contact_loss_neighborhood.py](bench_r35_contact_loss_neighborhood.py)
- [bench_r35_contact_mip.py](bench_r35_contact_mip.py)
- [bench_r35_contact_pages.py](bench_r35_contact_pages.py)
- [bench_r35_contact_phase_walk.py](bench_r35_contact_phase_walk.py)
- [bench_r35_contact_phases.py](bench_r35_contact_phases.py)
- [bench_r35_contact_sat_portfolio.py](bench_r35_contact_sat_portfolio.py)
- [bench_r35_contact_slack.py](bench_r35_contact_slack.py)
- [bench_r35_corner_chains.py](bench_r35_corner_chains.py)
- [bench_r35_correlation_models.py](bench_r35_correlation_models.py)
- [bench_r35_coverage_adverse.py](bench_r35_coverage_adverse.py)
- [bench_r35_coverage_packing.py](bench_r35_coverage_packing.py)
- [bench_r35_cube_queries.py](bench_r35_cube_queries.py)
- [bench_r35_cut_inventory.py](bench_r35_cut_inventory.py)
- [bench_r35_cycle_branches.py](bench_r35_cycle_branches.py)
- [bench_r35_cycle_contacts.py](bench_r35_cycle_contacts.py)
- [bench_r35_fixed_packing.py](bench_r35_fixed_packing.py)
- [bench_r35_forbidden_repair.py](bench_r35_forbidden_repair.py)
- [bench_r35_frame_defaults.py](bench_r35_frame_defaults.py)
- [bench_r35_frame_packet.py](bench_r35_frame_packet.py)
- [bench_r35_frame_tracking.py](bench_r35_frame_tracking.py)
- [bench_r35_geographic_board_lift.py](bench_r35_geographic_board_lift.py)
- [bench_r35_geographic_frontier.py](bench_r35_geographic_frontier.py)
- [bench_r35_geographic_portfolio.py](bench_r35_geographic_portfolio.py)
- [bench_r35_geographic_prices.py](bench_r35_geographic_prices.py)
- [bench_r35_geographic_replay.py](bench_r35_geographic_replay.py)
- [bench_r35_geographic_units.py](bench_r35_geographic_units.py)
- [bench_r35_gpu_pages.py](bench_r35_gpu_pages.py)
- [bench_r35_identity_duals.py](bench_r35_identity_duals.py)
- [bench_r35_integer_conflicts.py](bench_r35_integer_conflicts.py)
- [bench_r35_joint_residuals.py](bench_r35_joint_residuals.py)
- [bench_r35_loss_euler_split.py](bench_r35_loss_euler_split.py)
- [bench_r35_loss_euler_weights.py](bench_r35_loss_euler_weights.py)
- [bench_r35_loss_group_proof.py](bench_r35_loss_group_proof.py)
- [bench_r35_loss_mate_global.py](bench_r35_loss_mate_global.py)
- [bench_r35_loss_mates.py](bench_r35_loss_mates.py)
- [bench_r35_loss_overlap.py](bench_r35_loss_overlap.py)
- [bench_r35_loss_threshold_proof.py](bench_r35_loss_threshold_proof.py)
- [bench_r35_loss_transfer_streams.py](bench_r35_loss_transfer_streams.py)
- [bench_r35_loss_transfers.py](bench_r35_loss_transfers.py)
- [bench_r35_lower_loss_witness.py](bench_r35_lower_loss_witness.py)
- [bench_r35_mate_clauses.py](bench_r35_mate_clauses.py)
- [bench_r35_mate_correlations.py](bench_r35_mate_correlations.py)
- [bench_r35_mate_rollback.py](bench_r35_mate_rollback.py)
- [bench_r35_native_chunks.py](bench_r35_native_chunks.py)
- [bench_r35_orbit_archive.py](bench_r35_orbit_archive.py)
- [bench_r35_packing.py](bench_r35_packing.py)
- [bench_r35_phase_frontier.py](bench_r35_phase_frontier.py)
- [bench_r35_physical_adjacency.py](bench_r35_physical_adjacency.py)
- [bench_r35_plateau_controls.py](bench_r35_plateau_controls.py)
- [bench_r35_plateau_floors.py](bench_r35_plateau_floors.py)
- [bench_r35_plateau_frontier.py](bench_r35_plateau_frontier.py)
- [bench_r35_plateau_vectors.py](bench_r35_plateau_vectors.py)
- [bench_r35_portable_siblings.py](bench_r35_portable_siblings.py)
- [bench_r35_pose_contact_outcomes.py](bench_r35_pose_contact_outcomes.py)
- [bench_r35_pose_contact_starts.py](bench_r35_pose_contact_starts.py)
- [bench_r35_pose_contacts.py](bench_r35_pose_contacts.py)
- [bench_r35_pose_cuda.py](bench_r35_pose_cuda.py)
- [bench_r35_pose_literals.py](bench_r35_pose_literals.py)
- [bench_r35_pose_paths.py](bench_r35_pose_paths.py)
- [bench_r35_pose_seams.py](bench_r35_pose_seams.py)
- [bench_r35_pose_transactions.py](bench_r35_pose_transactions.py)
- [bench_r35_refit_cuts.py](bench_r35_refit_cuts.py)
- [bench_r35_remote_trajectories.py](bench_r35_remote_trajectories.py)
- [bench_r35_rotation_frames.py](bench_r35_rotation_frames.py)
- [bench_r35_rotation_vertex_response.py](bench_r35_rotation_vertex_response.py)
- [bench_r35_route_advanced.py](bench_r35_route_advanced.py)
- [bench_r35_route_budgets.py](bench_r35_route_budgets.py)
- [bench_r35_route_chains.py](bench_r35_route_chains.py)
- [bench_r35_route_contacts.py](bench_r35_route_contacts.py)
- [bench_r35_route_dependencies.py](bench_r35_route_dependencies.py)
- [bench_r35_route_layers.py](bench_r35_route_layers.py)
- [bench_r35_route_splits.py](bench_r35_route_splits.py)
- [bench_r35_shape_pages.py](bench_r35_shape_pages.py)
- [bench_r35_shell_overlap.py](bench_r35_shell_overlap.py)
- [bench_r35_signed_price_delta.py](bench_r35_signed_price_delta.py)
- [bench_r35_signed_refits.py](bench_r35_signed_refits.py)
- [bench_r35_speculative_poses.py](bench_r35_speculative_poses.py)
- [bench_r35_strand_matching.py](bench_r35_strand_matching.py)
- [bench_r35_tile_classes.py](bench_r35_tile_classes.py)
- [bench_r35_trajectories.py](bench_r35_trajectories.py)
- [bench_r35_turn_words.py](bench_r35_turn_words.py)
- [bench_r35_wide_undo.py](bench_r35_wide_undo.py)

## Evidence and source history

The [research reports](../docs/) and their JSON result/proof receipts retain the recorded evidence. Historical JSON filenames and hashes have not been rewritten: if a receipt names one of the old root paths, locate that same filename here. The [move manifest](relocation.json) maps every old path to its new path and records SHA-256 hashes before and after relocation. Two module docstrings have updated executable examples; their executable code is unchanged. All other moved drivers retain their original bytes.
