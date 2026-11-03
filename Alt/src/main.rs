//! Quantum Decoherence Simulator
//! A prototype engine for stateful, parallel-timeline NPC AI.
//! Inspired by Steins;Gate Attractor Fields and Hotline Miami combat pacing.

use macroquad::prelude::*;

// ==========================================
// SIMULATION & PHYSICS CONSTANTS
// ==========================================

const NUM_CHANNELS: usize = 50;
const TICKS_PER_UPDATE: u32 = 10;
const UPDATES_PER_SEC: f32 = 6.0;
const MIN_BASELINE_TICKS: usize = 1000;

// --- NPC MAPPING MATH ---
const NPC_DRIFT_SWEEP_RANGE: f32 = 35.0;
const ROOM_CENTER: f32 = 0.5;
const ROOM_USABLE_RADIUS: f32 = 0.46;
const NPC_MIN_SPEED: f32 = 0.006;
const NPC_SPEED_VARIANCE: f32 = 0.04;

// ==========================================
// DATA STRUCTURES
// ==========================================

#[derive(Clone, Copy)]
struct Bounds {
    min: f32,
    max: f32,
}

#[derive(Clone)]
struct QuantumState {
    tick_index: usize,
    values: [f32; NUM_CHANNELS],
    momentums: [f32; NUM_CHANNELS],
    // Cached variables for instant rendering
    composite_avg: f32,
    composite_x: f32, // Average of channels 0..25 (Used for 3D Phase Space)
    composite_y: f32, // Average of channels 25..50 (Used for 3D Phase Space)
    npc_pos: Vec2,
    npc_target: Vec2,
}

impl QuantumState {
    fn average_value(&self) -> f32 {
        self.values.iter().sum::<f32>() / NUM_CHANNELS as f32
    }
}

struct Worldline {
    id: usize,
    parent_id: Option<usize>, 
    branch_tick: usize,       
    abandoned_tick: Option<usize>,
    states: Vec<QuantumState>,     
    divergence_angle: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    SingleChannel(usize),
    CompositeAverage,
}

#[derive(Clone, Copy, PartialEq)]
enum GraphMode {
    Physical,
    GitTree,
    PhaseSpace3D,
}

// ==========================================
// SIMULATION ENGINE
// ==========================================

struct Simulation {
    bounds: [Bounds; NUM_CHANNELS],
    initial_values: [f32; NUM_CHANNELS],
    last_generated_timeline: Vec<QuantumState>,
    baseline_states: Vec<QuantumState>,
    archived_worldlines: Vec<Worldline>,
    active_worldline: Worldline,
    current_tick: usize,
    max_ticks: usize,
    
    // UI & State Toggles
    paused: bool,
    show_npc_window: bool,
    graph_mode: GraphMode, 
    frame_counter: u32,
    view_mode: ViewMode,
    
    // 3D Camera Controls
    cam_pitch: f32,
    cam_yaw: f32,

    // Live-tunable Engine Parameters
    max_step: f32,             
    half_life_secs: f32,       
    start_gravity_weight: f32, 
    current_gravity_weight: f32,
}

impl Simulation {
    fn new(
        max_step: f32,
        half_life_secs: f32,
        start_gravity_weight: f32,
        max_ticks: usize,
        view_mode: ViewMode,
        show_npc_window: bool,
        graph_mode: GraphMode,
    ) -> Self {
        rand::srand(miniquad::date::now() as u64);

        let mut initial_values = [0.0; NUM_CHANNELS];
        let mut bounds = [Bounds { min: 0.0, max: 0.0 }; NUM_CHANNELS];
        let mut init_sum = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for i in 0..NUM_CHANNELS {
            let start_val = rand::gen_range(0, 100) as f32;
            initial_values[i] = start_val;
            init_sum += start_val;
            if i < 25 { sum_x += start_val; } else { sum_y += start_val; }
            
            bounds[i] = Bounds {
                min: start_val - 100.0,
                max: start_val + 100.0,
            };
        }

        let t1_state = QuantumState {
            tick_index: 1,
            values: initial_values,
            momentums: [0.0; NUM_CHANNELS],
            composite_avg: init_sum / NUM_CHANNELS as f32,
            composite_x: sum_x / 25.0,
            composite_y: sum_y / 25.0,
            npc_pos: vec2(ROOM_CENTER, ROOM_CENTER),
            npc_target: vec2(ROOM_CENTER, ROOM_CENTER),
        };

        let mut baseline_states = vec![t1_state];
        for _ in 2..=max_ticks {
            let prev = baseline_states.last().unwrap();
            let next_s = Self::compute_next_state(&bounds, max_step, prev, None, 0.0);
            baseline_states.push(next_s);
        }

        let active_worldline = Worldline {
            id: 0,
            parent_id: None,
            branch_tick: 1,
            abandoned_tick: None,
            states: baseline_states.clone(),
            divergence_angle: 0.0,
        };

        Self {
            bounds,
            initial_values,
            last_generated_timeline: baseline_states.clone(),
            baseline_states,
            archived_worldlines: Vec::new(),
            active_worldline,
            current_tick: 1,
            max_ticks,
            paused: false,
            show_npc_window,
            graph_mode,
            frame_counter: 0,
            view_mode,
            cam_pitch: 0.4,
            cam_yaw: -0.6, 
            max_step,
            half_life_secs,
            start_gravity_weight,
            current_gravity_weight: 0.0,
        }
    }

    fn compute_next_state(
        bounds: &[Bounds; NUM_CHANNELS],
        max_step: f32,
        prev_state: &QuantumState,
        ghost_state: Option<&QuantumState>,
        gravity_w: f32,
    ) -> QuantumState {
        let next_tick = prev_state.tick_index + 1;
        let mut new_values = [0.0; NUM_CHANNELS];
        let mut new_momentums = [0.0; NUM_CHANNELS];
        let mut composite_sum = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for i in 0..NUM_CHANNELS {
            let current = prev_state.values[i];
            let momentum = prev_state.momentums[i];
            let b = bounds[i];

            let step_roll = rand::gen_range(-max_step, max_step);
            let roll = (current + step_roll).clamp(b.min, b.max);

            let goes_against = (momentum > 0.0 && roll < current) || (momentum < 0.0 && roll > current);

            let modified_roll = if goes_against {
                (roll + momentum).clamp(b.min, b.max)
            } else {
                (roll + momentum * 0.5).clamp(b.min, b.max)
            };

            let next_val = if let Some(ghost) = ghost_state {
                (current + modified_roll + gravity_w * ghost.values[i]) / (2.0 + gravity_w)
            } else {
                (current + modified_roll) / 2.0
            };

            let next_val = next_val.clamp(b.min, b.max);
            new_values[i] = next_val;
            new_momentums[i] = next_val - current;
            
            composite_sum += next_val;
            if i < 25 { sum_x += next_val; } else { sum_y += next_val; }
        }
        
        let sweep_scale = std::f32::consts::PI / NPC_DRIFT_SWEEP_RANGE; 
        let base_x = bounds[0].min + 100.0;
        let base_y = bounds[1].min + 100.0;

        let target_x = ROOM_CENTER + ROOM_USABLE_RADIUS * ((new_values[0] - base_x) * sweep_scale).sin();
        let target_y = ROOM_CENTER + ROOM_USABLE_RADIUS * ((new_values[1] - base_y) * sweep_scale).sin();
        let npc_target = vec2(target_x, target_y);

        let b_speed = bounds[2];
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
            composite_avg: composite_sum / NUM_CHANNELS as f32,
            composite_x: sum_x / 25.0,
            composite_y: sum_y / 25.0,
            npc_pos,
            npc_target,
        }
    }

    fn step_playhead(&mut self) {
        if self.current_tick < self.max_ticks {
            self.current_tick += 1;
        }
    }

    fn rewind_to_tick(&mut self, target_tick: usize) {
        if target_tick >= self.current_tick { return; }

        let old_id = self.active_worldline.id;
        self.active_worldline.abandoned_tick = Some(self.current_tick);
        let mut new_states = self.active_worldline.states[..target_tick].to_vec();
        
        self.last_generated_timeline = self.active_worldline.states.clone();

        let finished_line = std::mem::replace(
            &mut self.active_worldline,
            Worldline {
                id: self.archived_worldlines.len() + 1,
                parent_id: Some(old_id),
                branch_tick: target_tick,
                abandoned_tick: None,
                states: Vec::new(),
                divergence_angle: 0.0,
            },
        );
        self.archived_worldlines.push(finished_line);

        let half_life_ticks = (self.half_life_secs * UPDATES_PER_SEC).max(1.0);
        
        for t in target_tick..self.max_ticks {
            let prev = new_states.last().unwrap();
            let ghost = self.last_generated_timeline.get(t); 
            
            let ticks_since_jump = (t - target_tick) as f32;
            let gravity_w = self.start_gravity_weight * 0.5_f32.powf(ticks_since_jump / half_life_ticks);

            let next_s = Self::compute_next_state(&self.bounds, self.max_step, prev, ghost, gravity_w);
            new_states.push(next_s);
        }

        self.active_worldline.states = new_states;
        self.recalc_active_divergence();
        self.current_tick = target_tick;
    }

    fn recalc_active_divergence(&mut self) {
        let mut dot_product = 0.0;
        let mut mag_sq_c = 0.0; 
        let mut mag_sq_b = 0.0; 

        if self.baseline_states.is_empty() {
            self.active_worldline.divergence_angle = 0.0;
            return;
        }

        let initial = &self.initial_values;

        for t in 0..self.max_ticks {
            let active_s = &self.active_worldline.states[t];
            let base_s = &self.baseline_states[t];
            for c in 0..NUM_CHANNELS {
                let dc = active_s.values[c] - initial[c];
                let db = base_s.values[c] - initial[c];
                
                dot_product += dc * db;
                mag_sq_c += dc * dc;
                mag_sq_b += db * db;
            }
        }

        let mag_c = mag_sq_c.sqrt().max(0.00001); 
        let mag_b = mag_sq_b.sqrt().max(0.00001);
        let cos_theta = (dot_product / (mag_c * mag_b)).clamp(-1.0, 1.0);

        self.active_worldline.divergence_angle = cos_theta.acos().to_degrees();
    }

    fn adjust_max_ticks(&mut self, delta_ticks: i32) {
        let new_max = (self.max_ticks as i32 + delta_ticks).max(60) as usize; 
        if new_max == self.max_ticks { return; }
        
        let old_max = self.max_ticks;
        self.max_ticks = new_max;

        if new_max < old_max {
            self.baseline_states.truncate(new_max);
            self.last_generated_timeline.truncate(new_max);
            self.active_worldline.states.truncate(new_max);
            
            if self.current_tick > new_max { self.current_tick = new_max; }
            
            for wl in &mut self.archived_worldlines {
                wl.states.truncate(new_max);
                if let Some(abandoned) = wl.abandoned_tick {
                    if abandoned > new_max { wl.abandoned_tick = Some(new_max); }
                }
            }
            self.recalc_active_divergence();
        } else {
            while self.baseline_states.len() < new_max {
                let prev = self.baseline_states.last().unwrap().clone();
                let next_s = Self::compute_next_state(&self.bounds, self.max_step, &prev, None, 0.0);
                self.baseline_states.push(next_s);
            }
            
            while self.last_generated_timeline.len() < new_max {
                let prev = self.last_generated_timeline.last().unwrap().clone();
                let next_s = Self::compute_next_state(&self.bounds, self.max_step, &prev, None, 0.0);
                self.last_generated_timeline.push(next_s);
            }

            let target_tick = self.active_worldline.branch_tick;
            let half_life_ticks = (self.half_life_secs * UPDATES_PER_SEC).max(1.0);
            
            while self.active_worldline.states.len() < new_max {
                let prev = self.active_worldline.states.last().unwrap().clone();
                let next_idx = self.active_worldline.states.len(); 
                let ghost = self.last_generated_timeline.get(next_idx);
                let ticks_since_jump = (next_idx as f32 - target_tick as f32).max(0.0);
                let gravity_w = self.start_gravity_weight * 0.5_f32.powf(ticks_since_jump / half_life_ticks);
                
                let next_s = Self::compute_next_state(&self.bounds, self.max_step, &prev, ghost, gravity_w);
                self.active_worldline.states.push(next_s);
            }

            for wl in &mut self.archived_worldlines {
                while wl.states.len() < new_max {
                    let prev = wl.states.last().unwrap().clone();
                    let next_s = Self::compute_next_state(&self.bounds, self.max_step, &prev, None, 0.0);
                    wl.states.push(next_s);
                }
            }
            self.recalc_active_divergence();
        }
    }

    fn extract_y_value(&self, state: &QuantumState) -> f32 {
        match self.view_mode {
            ViewMode::CompositeAverage => state.composite_avg,
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

// ==========================================
// MAIN LOOP & UI RENDERER
// ==========================================

#[macroquad::main("Quantum Decoherence Simulator")]
async fn main() {
    let mut sim = Simulation::new(5.0, 6.0, 3.75, 1080, ViewMode::SingleChannel(0), true, GraphMode::Physical);
    let mut last_mouse_pos = vec2(0.0, 0.0);

    loop {
        clear_background(Color::from_rgba(14, 16, 22, 255));

        // --- INPUT HANDLING ---
        if is_key_pressed(KeyCode::Space) { sim.paused = !sim.paused; }
        if is_key_pressed(KeyCode::N) { sim.show_npc_window = !sim.show_npc_window; }
        if is_key_pressed(KeyCode::M) { 
            sim.graph_mode = match sim.graph_mode {
                GraphMode::Physical => GraphMode::GitTree,
                GraphMode::GitTree => GraphMode::PhaseSpace3D,
                GraphMode::PhaseSpace3D => GraphMode::Physical,
            }; 
        }
        if is_key_pressed(KeyCode::R) {
            sim = Simulation::new(sim.max_step, sim.half_life_secs, sim.start_gravity_weight, sim.max_ticks, sim.view_mode, sim.show_npc_window, sim.graph_mode);
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
        if is_key_pressed(KeyCode::F) { sim.adjust_max_ticks(-60); }
        if is_key_pressed(KeyCode::G) { sim.adjust_max_ticks(60); }

        // 3D Camera Controls
        let (mx, my) = mouse_position();
        if is_mouse_button_pressed(MouseButton::Right) {
            last_mouse_pos = vec2(mx, my);
        }
        if is_mouse_button_down(MouseButton::Right) {
            let dx = mx - last_mouse_pos.x;
            let dy = my - last_mouse_pos.y;
            sim.cam_yaw += dx * 0.01;
            sim.cam_pitch += dy * 0.01;
            sim.cam_pitch = sim.cam_pitch.clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0);
            last_mouse_pos = vec2(mx, my);
        }

        if !sim.paused {
            sim.frame_counter += 1;
            if sim.frame_counter >= TICKS_PER_UPDATE {
                sim.frame_counter = 0;
                sim.step_playhead();
            }
        }

        let margin_left = 70.0;
        let margin_right = 350.0; 
        let margin_top = 85.0;
        let margin_bottom = 180.0; 

        let plot_w = (screen_width() - margin_left - margin_right).max(100.0);
        let plot_h = (screen_height() - margin_top - margin_bottom).max(100.0);

        let (y_min, y_max) = match sim.view_mode {
            ViewMode::CompositeAverage => (0.0, 100.0),
            ViewMode::SingleChannel(s) => (sim.bounds[s].min, sim.bounds[s].max),
        };

        let tick_to_x = |tick: usize| -> f32 { margin_left + ((tick as f32 - 1.0) / (sim.max_ticks as f32 - 1.0)) * plot_w };
        let val_to_y = |val: f32| -> f32 { margin_top + plot_h - ((val - y_min) / (y_max - y_min)).clamp(0.0, 1.0) * plot_h };
        let max_lanes = sim.archived_worldlines.len() + 1;
        let lane_to_y = |id: usize| -> f32 { margin_top + (id as f32 + 0.5) * (plot_h / max_lanes as f32) };

        let get_visual_parent = |mut pid: usize, branch_tick: usize| -> usize {
            while let Some(parent_wl) = sim.archived_worldlines.get(pid) {
                if parent_wl.branch_tick <= branch_tick { break; }
                if let Some(next_pid) = parent_wl.parent_id { pid = next_pid; } else { break; }
            }
            pid
        };

        let project_3d = |t: usize, val_x: f32, val_y: f32| -> Vec2 {
            let dx = val_x - 50.0;
            let dy = val_y - 50.0; 
            let dz = (t as f32 / sim.max_ticks as f32) * 100.0 - 50.0; 

            let x1 = dx * sim.cam_yaw.cos() - dz * sim.cam_yaw.sin();
            let z1 = dx * sim.cam_yaw.sin() + dz * sim.cam_yaw.cos();
            let y1 = dy * sim.cam_pitch.cos() - z1 * sim.cam_pitch.sin();

            let cx = margin_left + plot_w / 2.0;
            let cy = margin_top + plot_h / 2.0;
            
            let scale_x = plot_w / 140.0;
            let scale_y = plot_h / 140.0;

            vec2(cx + x1 * scale_x, cy - y1 * scale_y) 
        };

        draw_rectangle_lines(margin_left, margin_top, plot_w, plot_h, 1.5, Color::from_rgba(50, 55, 70, 255));

        if sim.graph_mode == GraphMode::PhaseSpace3D {
            let origin = project_3d(0, 50.0, 50.0);
            let end_x = project_3d(0, 100.0, 50.0); 
            let end_y = project_3d(0, 50.0, 100.0); 
            let end_z = project_3d(sim.max_ticks, 50.0, 50.0); 

            draw_line(origin.x, origin.y, end_x.x, end_x.y, 2.0, Color::from_rgba(255, 100, 100, 150));
            draw_text("X: Ch 1-25", end_x.x + 5.0, end_x.y, 16.0, Color::from_rgba(255, 100, 100, 150));
            
            draw_line(origin.x, origin.y, end_y.x, end_y.y, 2.0, Color::from_rgba(100, 255, 100, 150));
            draw_text("Y: Ch 26-50", end_y.x + 5.0, end_y.y, 16.0, Color::from_rgba(100, 255, 100, 150));
            
            draw_line(origin.x, origin.y, end_z.x, end_z.y, 2.0, Color::from_rgba(100, 150, 255, 150));
            draw_text("Z: Time", end_z.x + 5.0, end_z.y, 16.0, Color::from_rgba(100, 150, 255, 150));
        } else if sim.graph_mode == GraphMode::Physical {
            for i in 0..=4 {
                let frac = i as f32 / 4.0;
                let y = margin_top + plot_h * (1.0 - frac);
                let val = y_min + frac * (y_max - y_min);
                draw_line(margin_left, y, margin_left + plot_w, y, 1.0, Color::from_rgba(30, 35, 48, 255));
                draw_text(&format!("{:.0}", val), 18.0, y + 5.0, 18.0, GRAY);
            }
            
            let ph_x = tick_to_x(sim.current_tick);
            draw_line(ph_x, margin_top, ph_x, margin_top + plot_h, 1.5, Color::from_rgba(255, 255, 255, 60));
            draw_text(&format!("LIVE: T{}", sim.current_tick), ph_x + 5.0, margin_top + 15.0, 16.0, LIGHTGRAY);
        }

        let num_archived = sim.archived_worldlines.len();
        for (idx, wl) in sim.archived_worldlines.iter().enumerate() {
            let is_baseline = wl.id == 0;
            let is_most_recent = idx + 1 == num_archived;

            let line_color = if is_baseline { Color::from_rgba(0, 210, 255, 190) }
            else if is_most_recent { Color::from_rgba(255, 165, 40, 210) }
            else { Color::from_rgba(110, 120, 145, 110) };
            let thickness = if is_baseline || is_most_recent { 2.5 } else { 1.5 };

            if sim.graph_mode == GraphMode::GitTree {
                let y = lane_to_y(wl.id);
                let start_x = tick_to_x(wl.branch_tick);
                let end_tick = wl.abandoned_tick.unwrap_or(sim.max_ticks);
                let end_x = tick_to_x(end_tick);

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
                draw_text(&format!("{:.6}°", wl.divergence_angle), end_x + 8.0, y + 5.0, 18.0, line_color);
            } else if sim.graph_mode == GraphMode::Physical {
                let mut prev_pt: Option<(f32, f32)> = None;
                for state in &wl.states {
                    if state.tick_index < wl.branch_tick { continue; }
                    let curr_x = tick_to_x(state.tick_index);
                    let curr_y = val_to_y(sim.extract_y_value(state)); 
                    
                    if let Some((px, py)) = prev_pt {
                        draw_line(px, py, curr_x, curr_y, thickness, line_color);
                    }
                    prev_pt = Some((curr_x, curr_y));
                }
                
                let end_x = tick_to_x(sim.max_ticks);
                let end_y = val_to_y(sim.extract_y_value(wl.states.last().unwrap()));
                draw_circle(end_x, end_y, 4.0, line_color);
                draw_text(&format!("{:.6}°", wl.divergence_angle), end_x + 8.0, end_y + 5.0, 18.0, line_color);
            } else if sim.graph_mode == GraphMode::PhaseSpace3D {
                let mut prev_pt: Option<Vec2> = None;
                for state in &wl.states {
                    if state.tick_index < wl.branch_tick { continue; }
                    let pt = project_3d(state.tick_index, state.composite_x, state.composite_y);
                    if let Some(p) = prev_pt {
                        draw_line(p.x, p.y, pt.x, pt.y, thickness, line_color);
                    }
                    prev_pt = Some(pt);
                }
            }
        }

        let active_color = Color::from_rgba(70, 255, 130, 255); 
        let ancestor_color = Color::from_rgba(70, 255, 130, 85);
        let wl = &sim.active_worldline;

        if sim.graph_mode == GraphMode::GitTree {
            let y = lane_to_y(wl.id);
            let start_x = tick_to_x(wl.branch_tick);
            let end_x = tick_to_x(sim.current_tick); 

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
        } else if sim.graph_mode == GraphMode::Physical {
            let mut prev_pt: Option<(f32, f32)> = None;
            for state in &wl.states {
                let curr_x = tick_to_x(state.tick_index);
                let curr_y = val_to_y(sim.extract_y_value(state));
                
                if let Some((px, py)) = prev_pt {
                    if state.tick_index <= wl.branch_tick {
                        draw_line(px, py, curr_x, curr_y, 2.0, ancestor_color);
                    } else if state.tick_index > sim.current_tick {
                        draw_line(px, py, curr_x, curr_y, 1.5, Color::from_rgba(70, 255, 130, 100)); 
                    } else {
                        draw_line(px, py, curr_x, curr_y, 3.0, active_color);
                    }
                }
                prev_pt = Some((curr_x, curr_y));
            }
            let cx = tick_to_x(sim.current_tick);
            let cy = val_to_y(sim.extract_y_value(&wl.states[sim.current_tick - 1]));
            draw_circle(cx, cy, 5.5, active_color);
        } else if sim.graph_mode == GraphMode::PhaseSpace3D {
            let mut prev_pt: Option<Vec2> = None;
            for state in &wl.states {
                let pt = project_3d(state.tick_index, state.composite_x, state.composite_y);
                if let Some(p) = prev_pt {
                    if state.tick_index <= wl.branch_tick {
                        draw_line(p.x, p.y, pt.x, pt.y, 2.0, ancestor_color);
                    } else if state.tick_index > sim.current_tick {
                        draw_line(p.x, p.y, pt.x, pt.y, 1.5, Color::from_rgba(70, 255, 130, 100)); 
                    } else {
                        draw_line(p.x, p.y, pt.x, pt.y, 3.0, active_color);
                    }
                }
                prev_pt = Some(pt);
            }
            let head_s = &wl.states[sim.current_tick - 1];
            let pt = project_3d(head_s.tick_index, head_s.composite_x, head_s.composite_y);
            draw_circle(pt.x, pt.y, 5.5, active_color);
        }

        let mut hovered_rewind_tick: Option<usize> = None;

        if sim.paused && sim.graph_mode != GraphMode::PhaseSpace3D {
            let (mx, my) = mouse_position();
            if mx >= margin_left && mx <= margin_left + plot_w && my >= margin_top && my <= margin_top + plot_h {
                
                if let Some(closest_state) = sim.active_worldline.states[..sim.current_tick].iter().min_by(|a, b| {
                    let da = (tick_to_x(a.tick_index) - mx).abs();
                    let db = (tick_to_x(b.tick_index) - mx).abs();
                    da.partial_cmp(&db).unwrap()
                }) {
                    let target_tick = closest_state.tick_index;
                    hovered_rewind_tick = Some(target_tick);
                    
                    let cx = tick_to_x(target_tick);
                    let cy = if sim.graph_mode == GraphMode::GitTree {
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

        let display_tick = hovered_rewind_tick.unwrap_or(sim.current_tick);

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
            if let Some(_) = sim.archived_worldlines.last() {
                let ghost_color = Color::from_rgba(255, 165, 40, 255);
                draw_agent(&sim.last_generated_timeline, ghost_color, Color::from_rgba(255, 165, 40, 110), 6.5, true);
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
        let ghost_state_opt = sim.last_generated_timeline.get(display_idx);

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

        let mut status_str = if sim.paused { "PAUSED (Click graph to Rewind | SPACE: Resume | M: View | N: NPC)".to_string() }
        else { "RUNNING (SPACE: Pause & Rewind | M: View Mode | N: NPC Arena)".to_string() };
        
        if sim.graph_mode == GraphMode::PhaseSpace3D {
            status_str = format!("{} | Right-Click: Rotate 3D Cube", status_str);
        }
        
        let status_col = if sim.paused { YELLOW } else { GREEN };
        draw_text(&status_str, margin_left, 30.0, 22.0, status_col);

        let total_secs = sim.max_ticks as f32 / UPDATES_PER_SEC;
        let mins = (total_secs / 60.0).floor() as u32;
        let secs = (total_secs % 60.0) as u32;
        let time_str = format!("{}:{:02}", mins, secs);

        let view_str = match sim.view_mode {
            ViewMode::SingleChannel(s) => format!(
                "Ch #{:02}/50 | Step: ±{:.0} (W/S) | Half-Life: {:.0}s (Q/E) | Pull: {:.0}% (Z/X) | Time: {} (F/G)",
                s + 1, sim.max_step, sim.half_life_secs, sim.start_gravity_weight * 100.0, time_str
            ),
            ViewMode::CompositeAverage => format!(
                "Composite (50 Ch) | Step: ±{:.0} (W/S) | Half-Life: {:.0}s (Q/E) | Pull: {:.0}% (Z/X) | Time: {} (F/G)",
                sim.max_step, sim.half_life_secs, sim.start_gravity_weight * 100.0, time_str
            ),
        };
        draw_text(&view_str, margin_left, 56.0, 17.0, LIGHTGRAY);

        draw_text("Cyan: 0.0° Fate | Amber: Run N-1 | Green: Active | C/A/D: Ch | M: Switch View | N: NPC | R: Reset", margin_left, screen_height() - 15.0, 17.5, GRAY);

        let meter_w = 320.0;
        let meter_h = 65.0;
        let meter_x = screen_width() - meter_w - 25.0;
        let meter_y = screen_height() - meter_h - 15.0; 

        draw_rectangle(meter_x, meter_y, meter_w, meter_h, Color::from_rgba(22, 14, 10, 240));
        draw_rectangle_lines(meter_x, meter_y, meter_w, meter_h, 2.0, Color::from_rgba(255, 120, 30, 200));
        draw_text("CURRENT RUN DIVERGENCE", meter_x + 14.0, meter_y + 22.0, 16.0, Color::from_rgba(200, 130, 80, 255));

        let meter_val = format!("{:.6}°", sim.active_worldline.divergence_angle);
        draw_text(&meter_val, meter_x + 14.0, meter_y + 52.0, 30.0, Color::from_rgba(255, 145, 35, 255));

        next_frame().await
    }
}