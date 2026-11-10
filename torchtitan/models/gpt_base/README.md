# GptBase

A decoder that exists for test coverage, not for training a released model. It
has no checkpoint and no state-dict adapter.

One `debugmodel` flavor mixes every shared layer type, repeating a four-layer
pattern so every stage of a 2- or 4-stage pipeline split sees each type:

| layer % 4 | attention | FFN |
|---|---|---|
| 0 | GQA (CosSin RoPE, QK norm) | dense MLP |
| 1 | MLA, no query LoRA | MoE, softmax router, routed experts only |
| 2 | GQA (CosSin RoPE, QK norm) | MoE, grouped sigmoid router, shared experts, aux loss |
| 3 | MLA, query LoRA, YaRN RoPE | dense MLP |

Sharding reuses Qwen3's per-layer sharding for GQA layers and DeepSeek V3's for
MLA layers. Parallelize reuses Qwen3's generic decoder path.

Integration tests live in `tests/integration_tests/models.py` (`gpt_base_*`)
and cover FSDP, TP+SP, CP (KV all-gather, ptrr load balancing), PP
(1F1B, Interleaved1F1B), EP, compile, selective AC, varlen attention, SPMD type
checking, and DistMuon (`config_registry.dist_muon_optimizer`).
