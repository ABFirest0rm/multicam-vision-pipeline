# Multi-Camera Real-Time Vision Pipeline

> 🚧 **Cleanup in progress.** This is the working research version, published as-is.
> Config, CLI, benchmarks and a demo GIF are being added — see [Roadmap](#roadmap).

A real-time video analytics pipeline that runs **detection → multi-object tracking (with ReID) →
face recognition** on multiple camera/video streams at once, built around a multi-process,
stage-based architecture so a slow stage never stalls the whole stream.

## What it does

- **Detection:** YOLO12n (Ultralytics) for people and vehicles, on GPU.
- **Tracking:** BoostTrack / BoT-SORT via `boxmot`, with OSNet (`osnet_x0_25_msmt17`) appearance ReID.
- **Face recognition:** InsightFace `buffalo_l` embeddings, cosine-matched against a local face DB,
  then **associated to tracks** so a name sticks to a person across frames.
- **Multi-camera:** one independent pipeline (set of processes) per source — webcam or video file.
- **Monitoring:** per-camera CPU / RAM / per-process GPU memory via `psutil` + NVML.

## Architecture

```
            per camera (one set of processes each)
 ┌─────────┐   q   ┌──────┐   q   ┌─────────┐   q   ┌─────────┐
 │ Capture │ ────▶ │ YOLO │ ────▶ │ Tracker │ ────▶ │ Display │
 │ (PyAV)  │       │      │       │ + ReID  │       │         │
 └─────────┘       └──────┘       └─────────┘       └──┬───▲──┘
                                                      q │   │ q
                                                   ┌────▼───┴────┐
                                                   │  Face rec + │
                                                   │ track match │
                                                   └─────────────┘
```

## Design decisions

- **Bounded queues with drop-oldest** (`frame_skip_fun.safe_put_to_queue`): under load the pipeline
  sheds stale frames instead of accumulating latency — for live video, a fresh frame beats a complete backlog.
- **Detection on alternate frames:** YOLO runs every other frame and the tracker carries identities
  through the gap — a deliberate throughput/accuracy trade-off.
- **Face↔track association** (`match_fbbox_to_tbbox_fn.py`): vectorized containment mask (faces inside
  person boxes), nearest-top-edge selection, and conflict resolution when two faces claim one track.
- **Process-per-stage** (not threads) to sidestep the GIL for CPU-heavy stages.

Observed ~20 FPS per stream on an RTX 4060 laptop GPU. A reproducible benchmark script is on the roadmap.

## Running (current state)

```bash
pip install -r requirements.txt
python main_multi_cam.py      # edit `sources` inside first
```

Paths to the face DB and video sources are currently hardcoded — moving to a config file is the first roadmap item.
`add_to_db.py` builds the face embedding DB from a folder of `person_name/*.jpg`.

## Known issues

- Hardcoded paths and camera names; no config/CLI yet.
- No coordinated shutdown across processes in multi-camera mode.

## Roadmap

- [ ] Config file + CLI (`--source`, `--tracker`, `--no-face`)
- [ ] Benchmark script: FPS and per-stage latency for 1 / 2 / 4 streams
- [ ] Clean shutdown, structured logging
- [ ] Demo GIF on public footage + architecture diagram

## Ethics

Face recognition identifies people. This project is for learning and research; use only footage you
have the right to process and faces of people who have consented to be in the database.
