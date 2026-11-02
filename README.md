# Quantum Decoherence Simulator 🕰️

A prototype engine for stateful, parallel-timeline NPC AI, inspired by *Steins;Gate* Attractor Fields and *Hotline Miami* combat pacing. Built in Rust using `macroquad`.

Instead of writing bespoke, hard-coded squad AI or using pure RNG, this engine generates 50 shared "Quantum Channels." NPCs base their movement, accuracy, and tactical decisions on these channels. When a player rewinds time, the timeline is pulled toward the "Ghost" of their previous attempt, allowing timelines to softly mirror each other before smoothly decohering into chaotic, butterfly-effect futures.

## Features

* **Worldline Gravity:** Time jumps seamlessly stitch timeline memories together. A tunable exponential half-life pulls new random rolls toward the ghost of your previous run, perfectly replicating the "Attractor Field" effect.
* **Deterministic AI Pendulum Mapping:** Channels map to 2D room coordinates using Sine-wave math. This translates abstract 1D quantum drift into smooth, wall-to-wall 2D patrol routes for NPCs.
* **51-Dimensional Divergence Meter:** Calculates the true geometric angle between the original "Fate" baseline and your current timeline using a 51-dimensional Cosine Similarity formula (50 spatial channels + 1 Time axis).
* **Live Interactive UI:** Features a real-time graphed timeline, a live NPC arena overlay, and a synchronized 50-channel data grid.

## How to Run

Make sure you have [Rust installed](https://rustup.rs/), then clone the repo and run:

```
bash
cargo run --release
```

## Controls
**Time Travel & UI**

   * **SPACE** - Pause / Resume the quantum engine.

   * **Left Click** (While Paused) - Click anywhere on the active timeline graph to time-leap and spawn a new branch.

   * **N** - Toggle the 2D NPC Arena Overlay (Watch the Active NPC and Timeline Ghosts move in real-time).

   * **R** - Reset the entire simulation (generates a new True Fate Baseline).

**View Modes (Graph & Data Grid)**

   * **C** - Toggle between the Composite View (Average of all 50 channels) and Single Channel View.

   * **UP / DOWN (or W/S / Arrows)** - Cycle through the 50 individual Quantum Channels.

**Live Engine Tuning**

   * **W / S** - Tune Step Size (How fast channels drift per tick).

   * **Q / E** - Tune Half-Life (How long the previous timeline's gravity holds onto the new timeline before decohering).

   * **Z / X** - Tune Start Pull (The initial strength of the gravity tether right after a time jump).


## Example

**Early Ticks**

  Here you can see that early on in the run, the NPC will follow the same general walking pattern with only slight deviance from each timeline.
  
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1768381d-bb98-4254-bc00-afe35401bebc" />

**Late Ticks**

  However, the further along time goes, the timelines drift further apart and the NPC starts taking different paths.
  
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/682ce121-dd90-4658-8e05-78690eb78e86" />


