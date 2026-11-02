//! Quantum Decoherence Simulator
//! A prototype engine for stateful, parallel-timeline NPC AI.
//! Inspired by Steins;Gate Attractor Fields and Hotline Miami combat pacing.

use macroquad::prelude::*;

// ==========================================
// SIMULATION & PHYSICS CONSTANTS
// ==========================================

/// Total number of independent quantum channels available for NPCs to read.
const NUM_CHANNELS: usize = 50;

/// The clock rate of the quantum engine (6 updates per second at 60 FPS).
const TICKS_PER_UPDATE: u32 = 10;
const UPDATES_PER_SEC: f32 = 6.0;

/// The number of ticks we silently pre-simulate to establish the 0.0° Fate Baseline.
const MIN_BASELINE_TICKS: usize = 1000;

// --- DIVERGENCE METER MATH ---
/// The 51st Dimension in our angle calculation! Acts as the "forward momentum of time".
/// Larger values keep the angle tighter (e.g., 0° - 15° range) for typical runs.
const TIME_AXIS_WEIGHT: f32 = 250.0;

// --- NPC MAPPING MATH ---
/// How many points of drift across a channel cause the NPC to sweep from the center to a wall.
const NPC_DRIFT_SWEEP_RANGE: f32 = 35.0;
/// The center of the 2D room (Normalized 0.0 - 1.0).
const ROOM_CENTER: f32 = 0.5;
/// The usable radius of the room (0.46 means the NPC bounces at 0.04 and 0.96).
const ROOM_USABLE_RADIUS: f32 = 0.46;
/// Base movement speed per tick (Normalized room scale).
const NPC_MIN_SPEED: f32 = 0.006;
/// Additional speed granted by a maxed-out Speed Channel.
const NPC_SPEED_VARIANCE: f32 = 0.04;

// ==========================================
// DATA STRUCTURES
// ==========================================

#[derive(Clone, Copy)]
struct Bounds {
    min: f32,
    max: f32,
}

/// A snapshot of the universe at a specific tick.
#[derive(Clone)]
struct QuantumState {
    tick_index: usize,
    values: [f32; NUM_CHANNELS],
    momentums: [f32; NUM_CHANNELS],
    // NPC visual coordinates (0.0..=1.0 normalized mapping)
    npc_pos: Vec2,
    npc_target: Vec2,
}

impl QuantumState {
    fn average_value(&self) -> f32 {
        self.values.iter().sum::<f32>() / NUM_CHANNELS as f32
    }
}

/// A continuous history of a specific gameplay attempt.
struct Worldline {
    id: usize,
    parent_id: Option<usize>, // Which timeline this branched from
    branch_tick: usize,       // The tick where this branch split from the previous run
    states: Vec<QuantumState>, // The played history (T1..=End of this run)
    divergence_angle: Option<f32>,
}

#[derive(Clone, Copy)]
enum ViewMode {
    SingleChannel(usize),
    CompositeAverage,
}

// ==========================================
// SIMULATION ENGINE
// ==========================================

struct Simulation {
    bounds: [Bounds; NUM_CHANNELS],
    initial_values: [f32; NUM_CHANNELS],

    // Stores the values generated at tick `T` on the PREVIOUS run (Ghost memory)
    last_generated_at_tick: Vec<Option<[f32; NUM_CHANNELS]>>,

    // The hidden 0.0° Fate Baseline simulated on the first rewind
    baseline_states: Vec<QuantumState>,

    archived_worldlines: Vec<Worldline>,
    active_worldline: Worldline,

    // UI & State Toggles
    paused: bool,
    show_npc_window: bool,
    git_tree_mode: bool, // Toggles standard values view vs Git-style branch tree
    frame_counter: u32,
    last_divergence_readout: Option<f32>,
    view_mode: ViewMode,
    
    // Live-tunable Engine Parameters
    max_step: f32,             // Max step size per tick
    half_life_secs: f32,       // How fast the Ghost timeline's gravity fades
    start_gravity_weight: f32, // Initial strength of the Ghost timeline's gravity
    current_gravity_weight: f32,
}

impl Simulation {
    fn new(
        max_step: f32,
        half_life_secs: f32,
        start_gravity_weight: f32,
        view_mode: ViewMode,
        show_npc_window: bool,
        git_tree_mode: bool,
    ) -> Self {
        rand::srand(miniquad::date::now() as u64);

        let mut initial_values = [0.0; NUM_CHANNELS];
        let mut bounds = [Bounds { min: 0.0, max: 0.0 }; NUM_CHANNELS];

        for i in 0..NUM_CHANNELS {
            let start_val = rand::gen_range(0, 100) as f32;
            initial_values[i] = start_val;
            bounds[i] = Bounds {
                min: start_val - 100.0,
                max: start_val + 100.0,
            };
        }

        let initial_target = vec2(ROOM_CENTER, ROOM_CENTER);
        let t1_state = QuantumState {
            tick_index: 1,
            values: initial_values,
            momentums: [0.0; NUM_CHANNELS],
            npc_pos: vec2(ROOM_CENTER, ROOM_CENTER),
            npc_target: initial_target,
        };

        let mut last_generated_at_tick = vec![None; MIN_BASELINE_TICKS + 500];
        last_generated_at_tick[1] = Some(initial_values);

        let active_worldline = Worldline {
            id: 0,
            parent_id: None,
            branch_tick: 1,
            states: vec![t1_state],
            divergence_angle: None,
        };

        Self {
            bounds,
            initial_values,
            last_generated_at_tick,
            baseline_states: Vec::new(),
            archived_worldlines: Vec::new(),
            active_worldline,
            paused: false,
            show_npc_window,
            git_tree_mode,
            frame_counter: 0,
            last_divergence_readout: None,
            view_mode,
            max_step,
            half_life_secs,
            start_gravity_weight,
            current_gravity_weight: 0.0,
        }
    }

    fn compute_next_state(
        &self,
        prev_state: &QuantumState,
        last_t_values: Option<[f32; NUM_CHANNELS]>,
        gravity_w: f32,
    ) -> QuantumState {
        let next_tick = prev_state.tick_index + 1;
        let mut new_values = [0.0; NUM_CHANNELS];
        let mut new_momentums = [0.0; NUM_CHANNELS];

        for i in 0..NUM_CHANNELS {
            let current = prev_state.values[i];
            let momentum = prev_state.momentums[i];
            let b = self.bounds[i];

            let step_roll = rand::gen_range(-self.max_step, self.max_step);
            let roll = (current + step_roll).clamp(b.min, b.max);

            let goes_against =
                (momentum > 0.0 && roll < current) || (momentum < 0.0 && roll > current);

            let modified_roll = if goes_against {
                (roll + momentum).clamp(b.min, b.max)
            } else {
                (roll + momentum * 0.5).clamp(b.min, b.max)
            };

            let next_val = if let Some(ghost_vals) = last_t_values {
                (current + modified_roll + gravity_w * ghost_vals[i]) / (2.0 + gravity_w)
            } else {
                (current + modified_roll) / 2.0
            };

            let next_val = next_val.clamp(b.min, b.max);

            new_values[i] = next_val;
            new_momentums[i] = next_val - current;
        }

        let start_x = self.initial_values[0];
        let start_y = self.initial_values[1];

        let sweep_scale = std::f32::consts::PI / NPC_DRIFT_SWEEP_RANGE; 
        let target_x = ROOM_CENTER + ROOM_USABLE_RADIUS * ((new_values[0] - start_x) * sweep_scale).sin();
        let target_y = ROOM_CENTER + ROOM_USABLE_RADIUS * ((new_values[1] - start_y) * sweep_scale).sin();
        let npc_target = vec2(target_x, target_y);

        let b_speed = self.bounds[2];
        let norm_speed = ((new_values[2] - b_speed.min) / (b_speed.max - b_speed.min)).clamp(0.0, 1.0);
        let speed = NPC_MIN_SPEED + norm_speed * NPC_SPEED_VARIANCE; 
        
        let to_target = npc_target - prev_state.npc_pos;
        let dist = to_target.length();

        let npc_pos = if dist > speed && dist > 0.0001 {
            prev_state.npc_pos + (to_target / dist) * speed
        } else {
            npc_target
        };

        QuantumState {
            tick_index: next_tick,
            values: new_values,
            momentums: new_momentums,
            npc_pos,
            npc_target,
        }
    }

    fn step_quantum_update(&mut self) {
        let prev_state = self.active_worldline.states.last().unwrap().clone();
        let next_tick = prev_state.tick_index + 1;

        if next_tick >= self.last_generated_at_tick.len() {
            self.last_generated_at_tick.resize(next_tick + 500, None);
        }

        let last_t_values = if self.archived_worldlines.is_empty() {
            None
        } else {
            self.last_generated_at_tick[next_tick]
        };

        let ticks_since_jump = next_tick.saturating_sub(self.active_worldline.branch_tick) as f32;
        let half_life_ticks = (self.half_life_secs * UPDATES_PER_SEC).max(1.0);
        let gravity_w = if last_t_values.is_some() {
            self.start_gravity_weight * 0.5_f32.powf(ticks_since_jump / half_life_ticks)
        } else {
            0.0
        };
        self.current_gravity_weight = gravity_w;

        let next_state = self.compute_next_state(&prev_state, last_t_values, gravity_w);

        self.last_generated_at_tick[next_tick] = Some(next_state.values);
        self.active_worldline.states.push(next_state);
    }

    fn rewind_to_tick(&mut self, target_tick: usize) {
        let Some(state_idx) = self
            .active_worldline
            .states
            .iter()
            .position(|s| s.tick_index == target_tick)
        else {
            return;
        };

        if state_idx + 1 >= self.active_worldline.states.len() {
            return;
        }

        let old_id = self.active_worldline.id;
        let inherited_history = self.active_worldline.states[..=state_idx].to_vec();
        let end_state = self.active_worldline.states.last().unwrap().clone();

        let clear_from = end_state.tick_index + 1;
        if clear_from < self.last_generated_at_tick.len() {
            for slot in self.last_generated_at_tick.iter_mut().skip(clear_from) {
                *slot = None;
            }
        }

        let angle_deg = if self.baseline_states.is_empty() {
            let mut silent_states = self.active_worldline.states.clone();
            while silent_states.last().unwrap().tick_index < MIN_BASELINE_TICKS {
                let prev = silent_states.last().unwrap();
                let next_s = self.compute_next_state(prev, None, 0.0);
                silent_states.push(next_s);
            }
            self.baseline_states = silent_states;
            0.0
        } else {
            let ref_idx = (end_state.tick_index - 1).min(self.baseline_states.len() - 1);
            let ref_baseline_state = &self.baseline_states[ref_idx];

            let mut dot_product = 0.0;
            let mut mag_sq_c = 0.0; 
            let mut mag_sq_b = 0.0; 

            for i in 0..NUM_CHANNELS {
                let dc = end_state.values[i] - self.initial_values[i];
                let db = ref_baseline_state.values[i] - self.initial_values[i];
                
                dot_product += dc * db;
                mag_sq_c += dc * dc;
                mag_sq_b += db * db;
            }

            let time_axis = TIME_AXIS_WEIGHT;
            dot_product += time_axis * time_axis;
            mag_sq_c += time_axis * time_axis;
            mag_sq_b += time_axis * time_axis;

            let mag_c = mag_sq_c.sqrt();
            let mag_b = mag_sq_b.sqrt();
            let cos_theta = (dot_product / (mag_c * mag_b)).clamp(-1.0, 1.0);

            cos_theta.acos().to_degrees()
        };

        self.active_worldline.divergence_angle = Some(angle_deg);
        self.last_divergence_readout = Some(angle_deg);
        self.current_gravity_weight = self.start_gravity_weight;

        let next_id = self.archived_worldlines.len() + 1;
        let finished_line = std::mem::replace(
            &mut self.active_worldline,
            Worldline {
                id: next_id,
                parent_id: Some(old_id),
                branch_tick: target_tick,
                states: inherited_history,
                divergence_angle: None,
            },
        );

        self.archived_worldlines.push(finished_line);
    }

    fn extract_y_value(&self, state: &QuantumState) -> f32 {
        match self.view_mode {
            ViewMode::CompositeAverage => state.average_value(),
            ViewMode::SingleChannel(ch) => state.values[ch],
        }
    }
}

// ==========================================
// HELPER FUNCTIONS
// ==========================================

fn get_interpolated_npc(
    states: &[QuantumState],
    display_tick: usize,
    sub_frame_alpha: f32,
) -> Option<(Vec2, Vec2)> {
    let idx = display_tick.checked_sub(1)?;
    if idx >= states.len() { return None; }
    let curr = &states[idx];
    if idx > 0 && sub_frame_alpha < 1.0 {
        let prev = &states[idx - 1];
        let pos = prev.npc_pos.lerp(curr.npc_pos, sub_frame_alpha);
        Some((pos, curr.npc_target))
    } else {
        Some((curr.npc_pos, curr.npc_target))
    }
}

fn draw_dashed_line(x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: Color, dash_len: f32, gap_len: f32) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist == 0.0 { return; }
    
    let dir_x = dx / dist;
    let dir_y = dy / dist;
    
    let mut current_dist = 0.0;
    while current_dist < dist {
        let start_x = x1 + dir_x * current_dist;
        let start_y = y1 + dir_y * current_dist;
        let next_dist = (current_dist + dash_len).min(dist);
        let end_x = x1 + dir_x * next_dist;
        let end_y = y1 + dir_y * next_dist;
        draw_line(start_x, start_y, end_x, end_y, thickness, color);
        current_dist = next_dist + gap_len;
    }
}

// ==========================================
// MAIN LOOP & UI RENDERER
// ==========================================

#[macroquad::main("Quantum Decoherence Simulator")]
async fn main() {
    let mut sim = Simulation::new(5.0, 6.0, 3.75, ViewMode::SingleChannel(0), true, false);

    loop {
        clear_background(Color::from_rgba(14, 16, 22, 255));

        if is_key_pressed(KeyCode::Space) { sim.paused = !sim.paused; }
        if is_key_pressed(KeyCode::N) { sim.show_npc_window = !sim.show_npc_window; }
        if is_key_pressed(KeyCode::M) { sim.git_tree_mode = !sim.git_tree_mode; }
        if is_key_pressed(KeyCode::R) {
            sim = Simulation::new(sim.max_step, sim.half_life_secs, sim.start_gravity_weight, sim.view_mode, sim.show_npc_window, sim.git_tree_mode);
        }
        if is_key_pressed(KeyCode::C) {
            sim.view_mode = match sim.view_mode {
                ViewMode::CompositeAverage => ViewMode::SingleChannel(0),
                ViewMode::SingleChannel(_) => ViewMode::CompositeAverage,
            };
        }
        if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
            sim.view_mode = match sim.view_mode {
                ViewMode::CompositeAverage => ViewMode::SingleChannel(0),
                ViewMode::SingleChannel(s) => ViewMode::SingleChannel((s + 1) % NUM_CHANNELS),
            };
        }
        if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
            sim.view_mode = match sim.view_mode {
                ViewMode::CompositeAverage => ViewMode::SingleChannel(NUM_CHANNELS - 1),
                ViewMode::SingleChannel(0) => ViewMode::SingleChannel(NUM_CHANNELS - 1),
                ViewMode::SingleChannel(s) => ViewMode::SingleChannel(s - 1),
            };
        }

        if is_key_pressed(KeyCode::S) { sim.max_step = (sim.max_step - 1.0).max(1.0); }
        if is_key_pressed(KeyCode::W) { sim.max_step = (sim.max_step + 1.0).min(50.0); }
        if is_key_pressed(KeyCode::Q) { sim.half_life_secs = (sim.half_life_secs - 1.0).max(1.0); }
        if is_key_pressed(KeyCode::E) { sim.half_life_secs = (sim.half_life_secs + 1.0).min(60.0); }
        if is_key_pressed(KeyCode::Z) { sim.start_gravity_weight = (sim.start_gravity_weight - 0.25).max(0.25); }
        if is_key_pressed(KeyCode::X) { sim.start_gravity_weight = (sim.start_gravity_weight + 0.25).min(10.0); }

        if !sim.paused {
            sim.frame_counter += 1;
            if sim.frame_counter >= TICKS_PER_UPDATE {
                sim.frame_counter = 0;
                sim.step_quantum_update();
            }
        }

        let margin_left = 70.0;
        let margin_right = 350.0; 
        let margin_top = 85.0;
        let margin_bottom = 180.0; 

        let plot_w = (screen_width() - margin_left - margin_right).max(100.0);
        let plot_h = (screen_height() - margin_top - margin_bottom).max(100.0);

        let max_tick = sim.archived_worldlines.iter()
            .chain(std::iter::once(&sim.active_worldline))
            .map(|w| w.states.last().unwrap().tick_index)
            .max().unwrap_or(50).max(50);

        let (y_min, y_max) = match sim.view_mode {
            ViewMode::CompositeAverage => (0.0, 100.0),
            ViewMode::SingleChannel(s) => (sim.bounds[s].min, sim.bounds[s].max),
        };

        let tick_to_x = |tick: usize| -> f32 { margin_left + ((tick as f32 - 1.0) / (max_tick as f32 - 1.0)) * plot_w };
        let val_to_y = |val: f32| -> f32 { margin_top + plot_h - ((val - y_min) / (y_max - y_min)).clamp(0.0, 1.0) * plot_h };
        
        let max_lanes = sim.archived_worldlines.len() + 1;
        let lane_to_y = |id: usize| -> f32 { margin_top + (id as f32 + 0.5) * (plot_h / max_lanes as f32) };

        // Helper: Recursively find the true visual parent lane for a branch point in Git Tree mode
        let get_visual_parent = |mut pid: usize, branch_tick: usize| -> usize {
            while let Some(parent_wl) = sim.archived_worldlines.get(pid) {
                if parent_wl.branch_tick <= branch_tick { break; }
                if let Some(next_pid) = parent_wl.parent_id { pid = next_pid; } else { break; }
            }
            pid
        };

        draw_rectangle_lines(margin_left, margin_top, plot_w, plot_h, 1.5, Color::from_rgba(50, 55, 70, 255));

        if !sim.git_tree_mode {
            for i in 0..=4 {
                let frac = i as f32 / 4.0;
                let y = margin_top + plot_h * (1.0 - frac);
                let val = y_min + frac * (y_max - y_min);
                draw_line(margin_left, y, margin_left + plot_w, y, 1.0, Color::from_rgba(30, 35, 48, 255));
                draw_text(&format!("{:.0}", val), 18.0, y + 5.0, 18.0, GRAY);
            }

            let last_jump_tick = sim.archived_worldlines.last().map(|w| w.states.last().unwrap().tick_index).unwrap_or(1);
            if !sim.baseline_states.is_empty() && last_jump_tick > 1 {
                let start_s = sim.baseline_states.first().unwrap();
                let ref_idx = (last_jump_tick - 1).min(sim.baseline_states.len() - 1);
                let end_s = &sim.baseline_states[ref_idx];
                let sx = tick_to_x(start_s.tick_index);
                let sy = val_to_y(sim.extract_y_value(start_s));
                let ex = tick_to_x(end_s.tick_index);
                let ey = val_to_y(sim.extract_y_value(end_s));
                draw_dashed_line(sx, sy, ex, ey, 2.0, Color::from_rgba(0, 210, 255, 100), 10.0, 8.0);
                draw_circle(ex, ey, 3.0, Color::from_rgba(0, 210, 255, 150));
            }
            if let Some(last_run) = sim.archived_worldlines.last() {
                let start_s = last_run.states.first().unwrap();
                let end_s = last_run.states.last().unwrap();
                let sx = tick_to_x(start_s.tick_index);
                let sy = val_to_y(sim.extract_y_value(start_s));
                let ex = tick_to_x(end_s.tick_index);
                let ey = val_to_y(sim.extract_y_value(end_s));
                draw_dashed_line(sx, sy, ex, ey, 2.0, Color::from_rgba(255, 165, 40, 100), 10.0, 8.0);
                draw_circle(ex, ey, 3.0, Color::from_rgba(255, 165, 40, 150));
            }
        }

        let num_archived = sim.archived_worldlines.len();
        for (idx, wl) in sim.archived_worldlines.iter().enumerate() {
            let is_baseline = wl.id == 0;
            let is_most_recent = idx + 1 == num_archived;

            let line_color = if is_baseline { Color::from_rgba(0, 210, 255, 190) }
            else if is_most_recent { Color::from_rgba(255, 165, 40, 210) }
            else { Color::from_rgba(110, 120, 145, 110) };
            let thickness = if is_baseline || is_most_recent { 2.5 } else { 1.5 };

            if sim.git_tree_mode {
                let y = lane_to_y(wl.id);
                let start_x = tick_to_x(wl.branch_tick);
                let end_x = tick_to_x(wl.states.last().unwrap().tick_index);

                if let Some(pid) = wl.parent_id {
                    let visual_pid = get_visual_parent(pid, wl.branch_tick);
                    let py = lane_to_y(visual_pid);
                    let branch_end = (start_x + 12.0).min(end_x); 
                    draw_line(start_x, py, branch_end, y, thickness, line_color);
                    draw_circle(start_x, py, 4.0, line_color);
                    draw_line(branch_end, y, end_x, y, thickness, line_color);
                } else {
                    draw_line(start_x, y, end_x, y, thickness, line_color);
                }

                draw_circle(end_x, y, 4.0, line_color);
                if let Some(angle) = wl.divergence_angle {
                    draw_text(&format!("{:.6}°", angle), end_x + 8.0, y + 5.0, 18.0, line_color);
                }
            } else {
                for pair in wl.states.windows(2) {
                    if pair[0].tick_index < wl.branch_tick { continue; }
                    let x1 = tick_to_x(pair[0].tick_index);
                    let y1 = val_to_y(sim.extract_y_value(&pair[0]));
                    let x2 = tick_to_x(pair[1].tick_index);
                    let y2 = val_to_y(sim.extract_y_value(&pair[1]));
                    draw_line(x1, y1, x2, y2, thickness, line_color);
                }
                if let Some(end_state) = wl.states.last() {
                    let end_x = tick_to_x(end_state.tick_index);
                    let end_y = val_to_y(sim.extract_y_value(end_state));
                    draw_circle(end_x, end_y, 4.0, line_color);
                    if let Some(angle) = wl.divergence_angle {
                        draw_text(&format!("{:.6}°", angle), end_x + 8.0, end_y + 5.0, 18.0, line_color);
                    }
                }
            }
        }

        let active_color = Color::from_rgba(70, 255, 130, 255); 
        let ancestor_color = Color::from_rgba(70, 255, 130, 85);
        let wl = &sim.active_worldline;

        if sim.git_tree_mode {
            let y = lane_to_y(wl.id);
            let start_x = tick_to_x(wl.branch_tick);
            let end_x = tick_to_x(wl.states.last().unwrap().tick_index);

            if let Some(pid) = wl.parent_id {
                let visual_pid = get_visual_parent(pid, wl.branch_tick);
                let py = lane_to_y(visual_pid);
                let branch_end = (start_x + 12.0).min(end_x);
                draw_line(start_x, py, branch_end, y, 3.0, active_color);
                draw_circle(start_x, py, 4.0, ancestor_color);
                draw_line(branch_end, y, end_x, y, 3.0, active_color);
            } else {
                draw_line(start_x, y, end_x, y, 3.0, active_color);
            }

            draw_circle(end_x, y, 5.5, active_color);
            let head_label = if sim.current_gravity_weight > 0.005 {
                format!("T{} ({:.0}% pull)", wl.states.last().unwrap().tick_index, sim.current_gravity_weight * 100.0)
            } else {
                format!("T{}", wl.states.last().unwrap().tick_index)
            };
            draw_text(&head_label, end_x + 8.0, y - 8.0, 18.0, active_color);
        } else {
            for pair in wl.states.windows(2) {
                let x1 = tick_to_x(pair[0].tick_index);
                let y1 = val_to_y(sim.extract_y_value(&pair[0]));
                let x2 = tick_to_x(pair[1].tick_index);
                let y2 = val_to_y(sim.extract_y_value(&pair[1]));

                if pair[0].tick_index < wl.branch_tick {
                    draw_line(x1, y1, x2, y2, 2.0, ancestor_color);
                } else {
                    draw_line(x1, y1, x2, y2, 3.0, active_color);
                }
            }
            if let Some(head) = wl.states.last() {
                let hx = tick_to_x(head.tick_index);
                let hy = val_to_y(sim.extract_y_value(head));
                draw_circle(hx, hy, 5.5, active_color);
                let head_label = if sim.current_gravity_weight > 0.005 {
                    format!("T{} ({:.0}% pull)", head.tick_index, sim.current_gravity_weight * 100.0)
                } else {
                    format!("T{}", head.tick_index)
                };
                draw_text(&head_label, hx + 8.0, hy - 8.0, 18.0, active_color);
            }
        }

        let mut hovered_rewind_tick: Option<usize> = None;

        if sim.paused {
            let (mx, my) = mouse_position();
            if mx >= margin_left && mx <= margin_left + plot_w && my >= margin_top && my <= margin_top + plot_h {
                if let Some(closest_state) = sim.active_worldline.states.iter().min_by(|a, b| {
                    let da = (tick_to_x(a.tick_index) - mx).abs();
                    let db = (tick_to_x(b.tick_index) - mx).abs();
                    da.partial_cmp(&db).unwrap()
                }) {
                    let target_tick = closest_state.tick_index;
                    hovered_rewind_tick = Some(target_tick);
                    
                    let cx = tick_to_x(target_tick);
                    let cy = if sim.git_tree_mode {
                        if target_tick < sim.active_worldline.branch_tick && sim.active_worldline.parent_id.is_some() {
                            lane_to_y(get_visual_parent(sim.active_worldline.parent_id.unwrap(), target_tick))
                        } else {
                            lane_to_y(sim.active_worldline.id)
                        }
                    } else {
                        val_to_y(sim.extract_y_value(closest_state))
                    };

                    draw_line(cx, margin_top, cx, margin_top + plot_h, 1.5, Color::from_rgba(255, 80, 80, 180));
                    draw_circle(cx, cy, 7.0, RED);
                    draw_text(&format!("Click to Rewind -> T{}", target_tick), (cx - 70.0).clamp(margin_left, margin_left + plot_w - 160.0), margin_top - 10.0, 20.0, RED);

                    if is_mouse_button_pressed(MouseButton::Left) {
                        sim.rewind_to_tick(target_tick);
                    }
                }
            }
        }

        let display_tick = hovered_rewind_tick.unwrap_or_else(|| sim.active_worldline.states.last().unwrap().tick_index);

        if sim.show_npc_window {
            let win_w = 420.0;
            let win_h = 420.0;
            let win_x = margin_left + 20.0;
            let win_y = margin_top + 20.0;
            let room_pad = 18.0;
            let room_x = win_x + room_pad;
            let room_y = win_y + 48.0;
            let room_w = win_w - room_pad * 2.0;
            let room_h = win_h - 48.0 - room_pad;

            draw_rectangle(win_x, win_y, win_w, win_h, Color::from_rgba(10, 12, 18, 235));
            draw_rectangle_lines(win_x, win_y, win_w, win_h, 2.0, Color::from_rgba(90, 110, 140, 255));
            draw_rectangle(room_x, room_y, room_w, room_h, Color::from_rgba(20, 24, 34, 255));
            
            for g in 1..4 {
                let frac = g as f32 / 4.0;
                draw_line(room_x + room_w * frac, room_y, room_x + room_w * frac, room_y + room_h, 1.0, Color::from_rgba(32, 38, 52, 255));
                draw_line(room_x, room_y + room_h * frac, room_x + room_w, room_y + room_h * frac, 1.0, Color::from_rgba(32, 38, 52, 255));
            }
            draw_rectangle_lines(room_x, room_y, room_w, room_h, 1.5, Color::from_rgba(65, 75, 95, 255));

            let sub_alpha = if sim.paused { 1.0 } else { (sim.frame_counter as f32 + 1.0) / TICKS_PER_UPDATE as f32 };

            draw_text(&format!("NPC ARENA @ T{} (N: Close)", display_tick), win_x + 14.0, win_y + 24.0, 19.0, WHITE);
            draw_text("Ch1:TargetX | Ch2:TargetY | Ch3:Speed", win_x + 14.0, win_y + 42.0, 15.0, GRAY);

            let to_screen = |norm: Vec2| -> Vec2 { vec2(room_x + norm.x * room_w, room_y + norm.y * room_h) };

            let draw_agent = |states: &[QuantumState], body_col: Color, trail_col: Color, radius: f32, show_target: bool| {
                let max_idx = display_tick.min(states.len());
                let trail_start = max_idx.saturating_sub(60); 
                if max_idx > 1 {
                    for pair in states[trail_start..max_idx].windows(2) {
                        let p1 = to_screen(pair[0].npc_pos);
                        let p2 = to_screen(pair[1].npc_pos);
                        draw_line(p1.x, p1.y, p2.x, p2.y, 2.0, trail_col);
                    }
                }
                if let Some((pos, target)) = get_interpolated_npc(states, display_tick, sub_alpha) {
                    let sp = to_screen(pos);
                    let st = to_screen(target);
                    if show_target {
                        draw_line(sp.x, sp.y, st.x, st.y, 1.0, Color::from_rgba((body_col.r * 255.0) as u8, (body_col.g * 255.0) as u8, (body_col.b * 255.0) as u8, 95));
                        draw_line(st.x - 5.0, st.y, st.x + 5.0, st.y, 1.5, body_col);
                        draw_line(st.x, st.y - 5.0, st.x, st.y + 5.0, 1.5, body_col);
                    }
                    draw_circle(sp.x, sp.y, radius, body_col);
                    draw_circle_lines(sp.x, sp.y, radius + 1.5, 1.2, BLACK);
                }
            };

            if !sim.baseline_states.is_empty() {
                draw_agent(&sim.baseline_states, Color::from_rgba(0, 210, 255, 140), Color::from_rgba(0, 210, 255, 65), 5.5, false);
            }
            if let Some(prev1) = sim.archived_worldlines.last() {
                let ghost_color = Color::from_rgba(255, 165, 40, 255);
                draw_agent(&prev1.states, ghost_color, Color::from_rgba(255, 165, 40, 110), 6.5, true);
            }
            draw_agent(&sim.active_worldline.states, active_color, Color::from_rgba(70, 255, 130, 150), 8.0, true);
        }

        draw_text(&format!("QUANTUM CHANNELS @ T{} (Active | Ghost N-1)", display_tick), margin_left, margin_top + plot_h + 18.0, 16.0, GRAY);
        
        let grid_y = margin_top + plot_h + 35.0;
        let col_w = plot_w / 10.0;
        let row_h = 22.0;
        let ghost_color = Color::from_rgba(255, 165, 40, 255);

        let display_idx = display_tick.saturating_sub(1);
        let active_state_opt = sim.active_worldline.states.get(display_idx);
        let ghost_state_opt = sim.archived_worldlines.last().and_then(|w| w.states.get(display_idx));

        for i in 0..NUM_CHANNELS {
            let row = i / 10;
            let col = i % 10;
            let cx = margin_left + col as f32 * col_w;
            let cy = grid_y + row as f32 * row_h;

            let is_npc = i < 3; 
            let mut x = cx;

            if is_npc { draw_text("[", x, cy, 16.0, LIGHTGRAY); }
            x += if is_npc { 7.0 } else { 4.0 };

            let a_str = active_state_opt.map_or("--".to_string(), |s| format!("{:.0}", s.values[i]));
            draw_text(&a_str, x, cy, 16.0, active_color);
            x += 26.0;

            draw_text("|", x, cy, 16.0, DARKGRAY);
            x += 8.0;

            let g_str = ghost_state_opt.map_or("--".to_string(), |s| format!("{:.0}", s.values[i]));
            draw_text(&g_str, x, cy, 16.0, ghost_color);
            x += 26.0;

            if is_npc { draw_text("]", x, cy, 16.0, LIGHTGRAY); }
        }

        let status_str = if sim.paused { "PAUSED (Click graph to Rewind | SPACE: Resume | M: View | N: NPC)" }
        else { "RUNNING (SPACE: Pause & Rewind | M: View Mode | N: NPC Arena)" };
        let status_col = if sim.paused { YELLOW } else { GREEN };
        draw_text(status_str, margin_left, 30.0, 22.0, status_col);

        let half_life_ticks = (sim.half_life_secs * UPDATES_PER_SEC) as usize;
        let view_str = match sim.view_mode {
            ViewMode::SingleChannel(s) => format!(
                "Ch #{:02}/50 [{:.0}..{:.0}] | Step(W/S): ±{:.0} | Half-Life(Q/E): {:.0}s ({}t) | Pull(Z/X): {:.0}%",
                s + 1, sim.bounds[s].min, sim.bounds[s].max, sim.max_step, sim.half_life_secs, half_life_ticks, sim.start_gravity_weight * 100.0
            ),
            ViewMode::CompositeAverage => format!(
                "Composite (50 Ch) | Step(W/S): ±{:.0} | Half-Life(Q/E): {:.0}s ({}t) | Pull(Z/X): {:.0}%",
                sim.max_step, sim.half_life_secs, half_life_ticks, sim.start_gravity_weight * 100.0
            ),
        };
        draw_text(&view_str, margin_left, 56.0, 17.0, LIGHTGRAY);

        draw_text("Cyan: 0.0° Fate | Amber: Run N-1 | Green: Active | C/A/D: Ch | M: Git View | N: NPC | R: Reset", margin_left, screen_height() - 15.0, 17.5, GRAY);

        let meter_w = 320.0;
        let meter_h = 65.0;
        let meter_x = screen_width() - meter_w - 25.0;
        let meter_y = screen_height() - meter_h - 15.0; 

        draw_rectangle(meter_x, meter_y, meter_w, meter_h, Color::from_rgba(22, 14, 10, 240));
        draw_rectangle_lines(meter_x, meter_y, meter_w, meter_h, 2.0, Color::from_rgba(255, 120, 30, 200));
        draw_text("LAST JUMP DIVERGENCE", meter_x + 14.0, meter_y + 22.0, 16.0, Color::from_rgba(200, 130, 80, 255));

        let meter_val = match sim.last_divergence_readout {
            Some(deg) => format!("{:.6}°", deg),
            None => "?.??????° (NO JUMP)".to_string(),
        };
        draw_text(&meter_val, meter_x + 14.0, meter_y + 52.0, 30.0, Color::from_rgba(255, 145, 35, 255));

        next_frame().await
    }
}