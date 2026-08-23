# How this fork diverges from upstream SAM3

`javrasya/sam3` is a fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3),
consumed by [Discern](https://github.com/javrasya/discern) as an editable sibling checkout rather
than as a release. It carries local changes that upstream does not have, and nothing in the code
itself says which is which unless we write it down. This file is that record, mandated by Discern
ADR 0002 ("Recording How Our SAM3 Fork Diverges From Upstream").

Two other mechanisms back it up, because a document nobody is forced to update decays:

* `git remote -v` has an `upstream` pointing at facebookresearch/sam3, so
  `git log upstream/main..HEAD` answers "what is local?" mechanically at any time. Add it with
  `git remote add upstream https://github.com/facebookresearch/sam3.git` in a fresh clone.
* Every locally added module, function and method carries the comment
  `# DISCERN FORK LOCAL ADDITION`, so a reader forms no assumption about upstream semantics at
  the point where they are actually reading the code.

The branch Discern consumes is `mps-support`, not `main`. `main` is near-upstream.

## Local divergences

| Divergence | Why it is here | Upstreamable? |
|---|---|---|
| **LLM-guided cropped propagation** — `sam3/model/sam3_unified_processor.py::propagate_with_llm_crop` and `sam3/agent/` | Small Objects are segmented poorly at full-frame resolution. Each frame is detected afresh inside a per-Object Zoom Window instead. It deliberately runs **without** the video tracker, which is the single most surprising thing about it: it is detection chained frame to frame, not tracking. | No — it is Discern's workflow, not a model capability. |
| **Apple Silicon (MPS) support** — device selection, lazy `decord` import | Development happens on Macs; upstream assumes CUDA. Corresponds to upstream PR #264. | Yes — it probably belongs upstream rather than here. |
| **Windows compatibility and agent error handling** | Path handling and error paths that crashed on Windows. | Yes, in principle. |
| **Hybrid video processor** | Mixes detector and tracker passes over a video. | Undecided. |
| **Tracker refactor** — parameters removed from `Sam3TrackerPredictor`, logging and empty-mask handling in `Sam3UnifiedProcessor` / `Sam3VideoBase` | Simplification made while debugging propagation; changes signatures upstream still has. | No — a local simplification, and a merge hazard. |
| **Packaging as a dependency** — `pyproject.toml`, config download from HF | Lets Discern depend on the fork from another `uv` project. | No. |

### Added by per-Object Zoom Anchors (Discern ADR 0001, spec javrasya/discern#91)

All five are new files with no upstream counterpart, and all are pure Python — no torch, no GPU,
no network — so they are testable without a model. `tests/` exists only for these.

| File | What it owns |
|---|---|
| `sam3/zoom_anchor.py` | Every decision about Zoom Window geometry and about the order frames are processed in: anchor precedence, the Zoom Lock and its guards, the tick cadence, how late a Re-grounding answer may be, and when an Object is too big for zooming to be worth anything. Shared by both consumers, so a fix reaches both. |
| `sam3/zoom_propagation.py` | Per-Object bookkeeping for one propagation pass: which Objects are still being propagated, the per-frame Zoom Anchor record, the absence streak. |
| `sam3/re_grounding.py` | The Re-grounding request/response contract: prompt, parsing, per-Object outcomes. Stdlib only. |
| `sam3/zoom_pass.py` | The Zoom Pass's own decisions: per-Object abandonment and honest per-mask outcomes. Called by Discern's `Sam3Backend.refine_all_frames`. |
| `sam3/agent/llm_crop_advisor.py` | The transport for Re-grounding: images, threads and a provider. Rewritten from a single combined request to one request per Object. |
