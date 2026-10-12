//! Iterative-deepening principal variation search.

pub mod ordering;
pub mod params;
pub mod timeman;
pub mod tt;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::Duration;

use crate::eval::evaluate;
use crate::movegen::{generate_all, generate_noisy, has_legal_move, is_legal, legal_moves};
use crate::moves::{Move, MoveList};
use crate::nnue::{self, Accumulator, Network, RefreshCache};
use crate::bitboard::Bitboard;
use crate::position::Position;
use crate::see::see_ge;
use crate::types::Color;
use ordering::*;
use params::*;
use timeman::{TimeManager, stop_scale};
use tt::{Bound, TT, TTHit};

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 32_001;
pub const MATE: i32 = 32_000;
/// Scores beyond this magnitude are mate scores.
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;
pub const DRAW: i32 = 0;
/// Static-eval placeholder for positions in check.
const NO_EVAL: i32 = -INF;
/// Frames before ply 0, so `ply - 4` lookups never underflow.
const STACK_OFFSET: usize = 4;

#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    /// Hard node limit: abort mid-iteration.
    pub nodes: Option<u64>,
    /// Soft node limit: don't start another iteration (used by datagen).
    pub soft_nodes: Option<u64>,
    pub movetime: Option<u64>,
    /// Remaining clock time per colour, in ms.
    pub time: [Option<u64>; 2],
    pub inc: [u64; 2],
    pub movestogo: Option<u64>,
    pub infinite: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

/// Per-ply search state.
#[derive(Clone, Copy)]
struct Frame {
    /// Corrected static eval, or `NO_EVAL` when in check.
    static_eval: i32,
    /// Piece-to key of the move being searched from this ply (`NO_KEY` for a
    /// null move or before the root).
    key: usize,
    /// Move excluded by a singular-extension verification search.
    excluded: Move,
    killers: [Move; 2],
    /// Beta cutoffs produced by nodes at this ply under the current parent.
    cutoffs: u32,
    /// Whether this node is or was part of a principal variation.
    tt_pv: bool,
}

impl Default for Frame {
    fn default() -> Self {
        Frame { static_eval: NO_EVAL, key: NO_KEY, excluded: Move::NULL, killers: [Move::NULL; 2], cutoffs: 0, tt_pv: false }
    }
}

/// Search state for one thread. Histories persist between moves of a game.
pub struct Worker {
    tt: Arc<TT>,
    stop: Arc<AtomicBool>,
    nodes: u64,
    seldepth: usize,
    root_depth: i32,
    limits: Limits,
    time: TimeManager,
    /// Butterfly history `[color][from * 64 + to]`.
    /// Butterfly history `[color * 4 + from_attacked * 2 + to_attacked][from * 64 + to]`.
    history: Vec<[i16; 4096]>,
    /// Continuation history `[previous piece-to][piece-to]`.
    cont: Vec<[i16; 768]>,
    /// Capture history `[piece-to][captured type]`.
    capture: Vec<[i16; 6]>,
    /// Pawn-structure eval correction `[color][pawn hash % CORR_SIZE]`.
    corr: Vec<[i16; CORR_SIZE]>,
    /// Non-pawn placement correction `[stm * 2 + colour][hash % CORR_SIZE]`.
    np_corr: Vec<[i16; CORR_SIZE]>,
    /// Eval correction by the last two moves `[piece-to two plies ago][piece-to last ply]`.
    cont_corr: Vec<[i16; 768]>,
    frames: Vec<Frame>,
    pv: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pv_len: [usize; MAX_PLY + 1],
    /// Hashes of every position from the game start to the current node.
    hashes: Vec<u64>,
    /// Entries before this root index are actual played game history.
    root_hash_index: usize,
    lmr: Box<[[i32; 64]; 64]>,
    /// Nodes spent under each root move (by from-to) in the current iteration.
    root_nodes: Vec<u64>,
    /// `None` when built without a network: material evaluation.
    net: Option<&'static Network>,
    /// Accumulator per ply, updated incrementally along the search path.
    acc: Vec<Accumulator>,
    /// Piece deltas for revisited king contexts, owned by this worker.
    refresh_cache: RefreshCache,
    pub silent: bool,
    pub chess960: bool,
    pub move_overhead: u64,
}

impl Worker {
    pub fn new(tt: Arc<TT>, stop: Arc<AtomicBool>) -> Worker {
        let lmr = Box::new([[0; 64]; 64]);
        let limits = Limits::default();
        let time = TimeManager::new(&limits, Color::White, 0);
        Worker {
            tt,
            stop,
            nodes: 0,
            seldepth: 0,
            root_depth: 0,
            limits,
            time,
            history: vec![[0; 4096]; 8],
            cont: vec![[0; 768]; 768],
            capture: vec![[0; 6]; 768],
            corr: vec![[0; CORR_SIZE]; 2],
            np_corr: vec![[0; CORR_SIZE]; 4],
            cont_corr: vec![[0; 768]; 768],
            frames: vec![Frame::default(); MAX_PLY + STACK_OFFSET + 2],
            pv: Box::new([[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1]),
            pv_len: [0; MAX_PLY + 1],
            hashes: Vec::with_capacity(1024),
            root_hash_index: 0,
            lmr,
            root_nodes: vec![0; 4096],
            net: nnue::network(),
            acc: vec![Accumulator::default(); MAX_PLY + 1],
            refresh_cache: RefreshCache::default(),
            silent: false,
            chess960: false,
            move_overhead: 10,
        }
    }

    fn rebuild_lmr(&mut self) {
        let base = LMR_BASE_X100.get() as f64 / 100.0;
        let divisor = LMR_DIVISOR_X100.get() as f64 / 100.0;
        for (depth, row) in self.lmr.iter_mut().enumerate().skip(1) {
            for (moves, r) in row.iter_mut().enumerate().skip(1) {
                *r = (base + (depth as f64).ln() * (moves as f64).ln() / divisor) as i32;
            }
        }
    }

    pub fn set_tt(&mut self, tt: Arc<TT>) {
        self.tt = tt;
    }

    /// Forget everything learned in the previous game.
    pub fn clear(&mut self) {
        self.history.iter_mut().for_each(|h| h.fill(0));
        self.cont.iter_mut().for_each(|h| h.fill(0));
        self.capture.iter_mut().for_each(|h| h.fill(0));
        self.corr.iter_mut().for_each(|h| h.fill(0));
        self.np_corr.iter_mut().for_each(|h| h.fill(0));
        self.cont_corr.iter_mut().for_each(|h| h.fill(0));
        self.frames.fill(Frame::default());
    }

    /// `game_hashes` lists the positions played so far, ending with `pos`.
    pub fn search(&mut self, pos: &Position, game_hashes: &[u64], limits: Limits) -> SearchResult {
        self.limits = limits;
        self.rebuild_lmr();
        self.time = TimeManager::new(&self.limits, pos.side_to_move(), self.move_overhead);
        self.nodes = 0;
        self.root_depth = 0;
        self.frames.fill(Frame::default());
        self.hashes.clear();
        self.hashes.extend_from_slice(game_hashes);
        if self.hashes.last() != Some(&pos.hash()) {
            self.hashes.push(pos.hash());
        }
        self.root_hash_index = self.hashes.len() - 1;
        self.tt.new_search();
        if let Some(net) = self.net {
            self.acc[0].refresh(net, pos);
        }

        let root_moves = legal_moves(pos);
        let mut result = SearchResult {
            best_move: root_moves.first().copied().unwrap_or(Move::NULL),
            score: 0,
            depth: 0,
            nodes: 0,
        };

        if !root_moves.is_empty() {
            let max_depth = self.limits.depth.unwrap_or(MAX_PLY as i32 - 1).clamp(1, MAX_PLY as i32 - 1);
            let mut stability = 0u32;
            for depth in 1..=max_depth {
                self.root_depth = depth;
                self.seldepth = 0;
                self.root_nodes.fill(0);
                let iteration_start = self.nodes;
                let score = self.aspiration(pos, depth, result.score);
                if self.aborted() {
                    // A root move that raised alpha was fully searched: keep it.
                    if self.pv_len[0] > 0 {
                        result.best_move = self.pv[0][0];
                    }
                    break;
                }
                let best = self.pv[0][0];
                let previous_score = result.score;
                stability = if best == result.best_move { stability + 1 } else { 0 };
                result = SearchResult { best_move: best, score, depth, nodes: self.nodes };
                if !self.silent {
                    self.report(depth, score);
                }
                let iteration_nodes = (self.nodes - iteration_start).max(1);
                let best_fraction = self.root_nodes[history_index(best)] as f64 / iteration_nodes as f64;
                // A falling score means trouble the shallower searches missed:
                // allow up to 50% more time to find a defence.
                let drop = (previous_score - score).clamp(0, 100);
                let falling = if depth >= 6 && drop >= 20 { 1.0 + drop as f64 / 200.0 } else { 1.0 };
                let scale = if depth >= 6 { stop_scale(stability, best_fraction) * falling } else { 1.0 };
                let node_limit = self.limits.nodes.into_iter().chain(self.limits.soft_nodes).min();
                if self.time.soft_exceeded_scaled(scale) || node_limit.is_some_and(|n| self.nodes >= n) {
                    break;
                }
                // A forced mate seen with depth to spare won't change.
                let mate_plies = MATE - score.abs();
                if !self.limits.infinite && score.abs() > MATE_BOUND && depth > 2 * mate_plies + 2 {
                    break;
                }
            }
        }

        // UCI: under `go infinite`, bestmove must wait for `stop`.
        if self.limits.infinite {
            while !self.stop.load(Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        result.nodes = self.nodes;
        result
    }

    fn report(&self, depth: i32, score: i32) {
        let ms = self.time.elapsed_ms();
        let nps = self.nodes * 1000 / ms.max(1);
        let score = if score.abs() > MATE_BOUND {
            let moves = if score > 0 { (MATE - score + 1) / 2 } else { -(MATE + score) / 2 };
            format!("mate {moves}")
        } else {
            format!("cp {score}")
        };
        let pv: Vec<String> = self.pv[0][..self.pv_len[0]].iter().map(|m| m.to_uci(self.chess960)).collect();
        println!(
            "info depth {depth} seldepth {} score {score} nodes {} nps {nps} hashfull {} time {ms} pv {}",
            self.seldepth,
            self.nodes,
            self.tt.hashfull(),
            pv.join(" ")
        );
    }

    fn aspiration(&mut self, pos: &Position, depth: i32, prev: i32) -> i32 {
        let mut delta = ASP_DELTA.get();
        let (mut alpha, mut beta) = if depth >= ASP_MIN_DEPTH {
            ((prev - delta).max(-INF), (prev + delta).min(INF))
        } else {
            (-INF, INF)
        };
        loop {
            let score = self.negamax(pos, alpha, beta, depth, 0, false);
            if self.aborted() {
                return score;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INF);
            } else if score >= beta {
                beta = (score + delta).min(INF);
            } else {
                return score;
            }
            delta += delta / 2;
        }
    }

    /// True once the search must unwind. The first iteration always completes
    /// so there is a legal best move to report.
    #[inline(always)]
    fn aborted(&self) -> bool {
        self.root_depth > 1 && self.stop.load(Relaxed)
    }

    #[inline(always)]
    fn check_limits(&mut self) {
        if self.root_depth <= 1 {
            return;
        }
        if self.limits.nodes.is_some_and(|n| self.nodes >= n) || (self.nodes & 2047 == 0 && self.time.hard_exceeded())
        {
            self.stop.store(true, Relaxed);
        }
    }

    // ------------------------------------------------------------ evaluation

    /// Uncorrected evaluation of `pos`, whose accumulator is `acc[ply]`.
    #[inline]
    fn raw_eval(&self, pos: &Position, ply: usize) -> i32 {
        match self.net {
            Some(net) => net.evaluate(&self.acc[ply], pos.side_to_move(), pos.occupied().count()),
            None => evaluate(pos),
        }
    }

    /// Raw eval adjusted by what searches of similar pawn structures found.
    #[inline]
    fn corrected_eval(&self, pos: &Position, raw: i32, ply: usize) -> i32 {
        let stm = pos.side_to_move().idx();
        let pawn = self.corr[stm][pos.pawn_hash() as usize % CORR_SIZE] as i32;
        let np: i32 = Color::ALL
            .iter()
            .map(|&c| self.np_corr[stm * 2 + c.idx()][pos.nonpawn_hash(c) as usize % CORR_SIZE] as i32)
            .sum();
        let cont = self.cont_corr_entry(ply).map_or(0, |(a, b)| self.cont_corr[a][b] as i32);
        (raw + (pawn + np / 2 + cont / 2) / CORR_GRAIN).clamp(-MATE_BOUND + 1, MATE_BOUND - 1)
    }

    /// Continuation-correction indices for the node at `ply`, when both
    /// preceding moves are real moves.
    #[inline]
    fn cont_corr_entry(&self, ply: usize) -> Option<(usize, usize)> {
        let f = ply + STACK_OFFSET;
        let (two, one) = (self.frames[f - 2].key, self.frames[f - 1].key);
        (two != NO_KEY && one != NO_KEY).then_some((two, one))
    }

    fn update_correction(&mut self, pos: &Position, raw: i32, best: i32, depth: i32, ply: usize) {
        let target = ((best - raw) * CORR_GRAIN).clamp(-CORR_MAX, CORR_MAX);
        let weight = (depth + 1).min(16);
        let stm = pos.side_to_move().idx();
        let blend = |entry: &mut i16| {
            let updated = (*entry as i32 * (256 - weight) + target * weight) / 256;
            *entry = updated.clamp(-CORR_MAX, CORR_MAX) as i16;
        };
        blend(&mut self.corr[stm][pos.pawn_hash() as usize % CORR_SIZE]);
        for c in Color::ALL {
            blend(&mut self.np_corr[stm * 2 + c.idx()][pos.nonpawn_hash(c) as usize % CORR_SIZE]);
        }
        if let Some((a, b)) = self.cont_corr_entry(ply) {
            blend(&mut self.cont_corr[a][b]);
        }
    }

    /// Plays `mv` and brings `acc[ply + 1]` up to date.
    #[inline]
    fn make_move(&mut self, pos: &Position, mv: Move, ply: usize) -> Position {
        let child = pos.make_move(mv);
        self.update_child_accumulator(pos, mv, &child, ply);
        child
    }

    /// Update only searched children; a check probe needs no NNUE work.
    #[inline]
    fn update_child_accumulator(&mut self, pos: &Position, mv: Move, child: &Position, ply: usize) {
        if let Some(net) = self.net {
            let (done, rest) = self.acc.split_at_mut(ply + 1);
            rest[0].update_from_cached(&done[ply], net, pos, mv, child, &mut self.refresh_cache);
        }
    }

    #[inline]
    fn make_null(&mut self, pos: &Position, ply: usize) -> Position {
        if self.net.is_some() {
            self.acc[ply + 1] = self.acc[ply];
        }
        pos.make_null()
    }

    fn is_draw(&self, pos: &Position) -> bool {
        if pos.halfmove_clock() >= 100 {
            // A mating move ends the game before the fifty-move draw.
            return !pos.in_check() || !legal_moves(pos).is_empty();
        }
        if pos.is_insufficient_material() {
            return true;
        }
        // Search-path cycles score as draws. In played history, require two
        // previous matches so the current position is the third occurrence.
        let n = self.hashes.len();
        let window = (pos.halfmove_clock() as usize).min(n - 1);
        let mut previous = 0;
        for back in (4..=window).step_by(2) {
            let index = n - 1 - back;
            if self.hashes[index] == pos.hash() {
                previous += 1;
                if index >= self.root_hash_index || previous == 2 {
                    return true;
                }
            }
        }
        false
    }

    fn update_pv(&mut self, ply: usize, mv: Move) {
        let child_len = self.pv_len[ply + 1];
        self.pv[ply][0] = mv;
        for i in 0..child_len {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child_len + 1;
    }

    // --------------------------------------------------------------- history

    #[inline]
    fn lmr_reduction(&self, depth: i32, moves: usize) -> i32 {
        self.lmr[(depth as usize).min(63)][moves.min(63)]
    }

    /// Combined butterfly + 1-ply + 2-ply continuation history of a quiet move.
    #[inline]
    fn quiet_history(&self, pos: &Position, mv: Move, ply: usize, threats: Bitboard) -> i32 {
        let f = ply + STACK_OFFSET;
        let key = piece_to_key(pos, mv);
        let mut score = self.history[butterfly_table(pos, mv, threats)][history_index(mv)] as i32;
        for back in [1, 2, 4] {
            let prev = self.frames[f - back].key;
            if prev != NO_KEY {
                score += self.cont[prev][key] as i32;
            }
        }
        score
    }

    fn score_moves(&self, pos: &Position, moves: &MoveList, tt_move: Move, ply: usize, threats: Bitboard) -> [i32; 256] {
        let killers = self.frames[ply + STACK_OFFSET].killers;
        let mut scores = [0; 256];
        for (i, &mv) in moves.iter().enumerate() {
            scores[i] = if mv == tt_move {
                TT_MOVE
            } else if is_noisy(mv) {
                let hist = if mv.is_capture() {
                    self.capture[piece_to_key(pos, mv)][captured_type(pos, mv).idx()] as i32
                } else {
                    0
                };
                let base = if see_ge(pos, mv, 0) { GOOD_NOISY } else { BAD_NOISY };
                base + noisy_gain(pos, mv) * 8 + hist / 16
            } else if mv.is_promotion() {
                UNDER_PROMO
            } else if mv == killers[0] {
                KILLER_0
            } else if mv == killers[1] {
                KILLER_1
            } else {
                self.quiet_history(pos, mv, ply, threats)
            };
        }
        scores
    }

    #[allow(clippy::too_many_arguments)]
    fn update_histories(
        &mut self,
        pos: &Position,
        best: Move,
        best_quiet: bool,
        quiets: &MoveList,
        noisies: &MoveList,
        depth: i32,
        ply: usize,
        threats: Bitboard,
    ) {
        let f = ply + STACK_OFFSET;
        let bonus = (HIST_BONUS_PER_DEPTH.get() * depth - HIST_BONUS_OFFSET.get()).min(HIST_BONUS_MAX.get());

        if best_quiet {
            let killers = &mut self.frames[f].killers;
            if killers[0] != best {
                killers[1] = killers[0];
                killers[0] = best;
            }
            let prev = [self.frames[f - 1].key, self.frames[f - 2].key, self.frames[f - 4].key];
            for &mv in quiets.iter() {
                let b = if mv == best { bonus } else { -bonus };
                apply_bonus(&mut self.history[butterfly_table(pos, mv, threats)][history_index(mv)], b, HIST_MAX);
                let key = piece_to_key(pos, mv);
                for p in prev {
                    if p != NO_KEY {
                        apply_bonus(&mut self.cont[p][key], b, HIST_MAX);
                    }
                }
            }
        }
        for &mv in noisies.iter() {
            if mv.is_capture() {
                let b = if mv == best { bonus } else { -bonus };
                apply_bonus(&mut self.capture[piece_to_key(pos, mv)][captured_type(pos, mv).idx()], b, HIST_MAX);
            }
        }
    }

    // ---------------------------------------------------------------- search

    fn negamax(&mut self, pos: &Position, mut alpha: i32, mut beta: i32, mut depth: i32, ply: usize, cut_node: bool) -> i32 {
        self.pv_len[ply] = 0;
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply);
        }

        let f = ply + STACK_OFFSET;
        let root = ply == 0;
        let pv_node = beta - alpha > 1;
        self.nodes += 1;
        self.check_limits();
        if self.aborted() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply + 1);

        let in_check = pos.in_check();
        let excluded = self.frames[f].excluded;
        let singular_search = !excluded.is_null();
        if !root {
            if self.is_draw(pos) {
                return DRAW;
            }
            // Stand-pat and selective cutoffs assume that a real move exists.
            // Establish terminal draws before using those estimates or the TT.
            if !in_check && !has_legal_move(pos) {
                return DRAW;
            }
            if ply >= MAX_PLY - 1 {
                return if in_check { DRAW } else { self.raw_eval(pos, ply) };
            }
            // Mate distance pruning: no line from here beats a shorter mate.
            alpha = alpha.max(-MATE + ply as i32);
            beta = beta.min(MATE - ply as i32 - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        self.frames[f + 1].cutoffs = 0;
        let tt_hit = if singular_search { None } else { self.tt.probe(pos.hash(), ply) };
        let tt_move = tt_hit.map_or(Move::NULL, |h| h.mv);
        let tt_pv = if singular_search { self.frames[f].tt_pv } else { pv_node || tt_hit.is_some_and(|h| h.pv) };
        self.frames[f].tt_pv = tt_pv;
        let tt_capture = !tt_move.is_null() && is_noisy(tt_move);
        if let Some(hit) = tt_hit
            && !pv_node
            && hit.depth >= depth
            && pos.halfmove_clock() < 90
            && tt_cutoff(&hit, alpha, beta)
        {
            return hit.score;
        }

        // Static evaluation: raw from the TT when available, corrected by
        // pawn-structure history, and sharpened by a usable TT bound.
        let (raw_eval, static_eval, eval) = if in_check {
            (NO_EVAL, NO_EVAL, NO_EVAL)
        } else if singular_search {
            let se = self.frames[f].static_eval;
            (se, se, se)
        } else {
            let raw = match tt_hit {
                Some(hit) if hit.eval != NO_EVAL => hit.eval,
                _ => self.raw_eval(pos, ply),
            };
            let se = self.corrected_eval(pos, raw, ply);
            let eval = match tt_hit {
                Some(hit) if pos.halfmove_clock() < 90 && hit.score.abs() < MATE_BOUND && tt_bound_tightens(&hit, se) => hit.score,
                _ => se,
            };
            (raw, se, eval)
        };
        self.frames[f].static_eval = static_eval;

        // Improving: better than our previous turn's static eval.
        let improving = !in_check && {
            let two = self.frames[f - 2].static_eval;
            let four = self.frames[f - 4].static_eval;
            if two != NO_EVAL { static_eval > two } else { four != NO_EVAL && static_eval > four }
        };

        // A quiet real move can reach a forced fifty-move draw despite a
        // winning static score. Null moves reset the virtual clock, so both
        // selective cutoffs need the same late-clock guard as TT estimates.
        if !pv_node && !in_check && !singular_search && pos.halfmove_clock() < 90 {
            // Reverse futility pruning: far above beta, assume it holds.
            if depth <= RFP_MAX_DEPTH.get()
                && beta.abs() < MATE_BOUND
                && eval - RFP_MARGIN.get() * (depth - improving as i32) >= beta
            {
                return eval;
            }

            // Null move pruning: if passing still fails high, a real move will too.
            if depth >= NMP_MIN_DEPTH
                && eval >= beta
                && static_eval >= beta
                && beta.abs() < MATE_BOUND
                && self.frames[f - 1].key != NO_KEY
                && pos.has_non_pawn_material(pos.side_to_move())
            {
                let r = NMP_BASE_REDUCTION.get() + depth / NMP_DEPTH_DIVISOR.get() + ((eval - beta) / NMP_EVAL_DIVISOR.get()).min(3);
                self.frames[f].key = NO_KEY;
                let child = self.make_null(pos, ply);
                self.hashes.push(child.hash());
                let score = -self.negamax(&child, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
                self.hashes.pop();
                if self.aborted() {
                    return 0;
                }
                if score >= beta {
                    return if score > MATE_BOUND { beta } else { score };
                }
            }

            // ProbCut: a good capture that beats beta by a wide margin in a
            // reduced search very likely beats beta at full depth too.
            let probcut_beta = beta + PROBCUT_MARGIN.get();
            if depth >= PROBCUT_MIN_DEPTH
                && beta.abs() < MATE_BOUND
                && !tt_hit.is_some_and(|h| h.depth >= depth - 3 && h.score < probcut_beta)
            {
                let mut moves = MoveList::new();
                generate_noisy(pos, &mut moves);
                let scores = self.score_moves(pos, &moves, tt_move, ply, Bitboard::EMPTY);
                let mut picker = MovePicker::new(moves, scores);
                while let Some((mv, _)) = picker.next() {
                    if !see_ge(pos, mv, probcut_beta - static_eval) {
                        continue;
                    }
                    self.frames[f].key = piece_to_key(pos, mv);
                    let child = self.make_move(pos, mv, ply);
                    self.hashes.push(child.hash());
                    let mut s = -self.qsearch(&child, -probcut_beta, -probcut_beta + 1, ply + 1);
                    if s >= probcut_beta {
                        s = -self.negamax(&child, -probcut_beta, -probcut_beta + 1, depth - 4, ply + 1, !cut_node);
                    }
                    self.hashes.pop();
                    if self.aborted() {
                        return 0;
                    }
                    if s >= probcut_beta {
                        self.tt.store(pos.hash(), mv, s, raw_eval, depth - 3, Bound::Lower, ply, tt_pv);
                        return s;
                    }
                }
            }
        }

        // Internal iterative reduction: without a TT move, ordering here is
        // poor; search shallower and let the next iteration fill the TT.
        if depth >= IIR_MIN_DEPTH && tt_move.is_null() && (pv_node || cut_node) && !singular_search {
            depth -= 1;
        }

        // Staged ordering: a legal TT move is searched before anything is
        // generated; the full list is built and scored only if it fails to
        // cut off.
        let threats = pos.threats();
        let tt_first = !tt_move.is_null() && tt_move != excluded && is_legal(pos, tt_move);
        let mut picker: Option<MovePicker> = None;
        if !tt_first {
            let mut moves = MoveList::new();
            generate_all(pos, &mut moves);
            if moves.is_empty() {
                return if in_check { -MATE + ply as i32 } else { DRAW };
            }
            let scores = self.score_moves(pos, &moves, tt_move, ply, threats);
            picker = Some(MovePicker::new(moves, scores));
        }
        let mut tt_pending = tt_first;

        let original_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = Move::NULL;
        let mut moves_searched = 0usize;
        let mut quiets_tried = MoveList::new();
        let mut noisies_tried = MoveList::new();
        let mut skip_quiets = false;
        let lmp_limit = ((LMP_BASE.get() + depth * depth) / (2 - improving as i32)) as usize;

        loop {
            let (mv, order_score) = if tt_pending {
                tt_pending = false;
                (tt_move, TT_MOVE)
            } else {
                if picker.is_none() {
                    let mut moves = MoveList::new();
                    generate_all(pos, &mut moves);
                    let scores = self.score_moves(pos, &moves, tt_move, ply, threats);
                    picker = Some(MovePicker::new(moves, scores));
                }
                match picker.as_mut().and_then(MovePicker::next) {
                    // The TT move was already searched in the first stage.
                    Some((mv, _)) if tt_first && mv == tt_move => continue,
                    Some(next) => next,
                    None => break,
                }
            };
            if mv == excluded {
                continue;
            }
            let quiet = !is_noisy(mv);
            if quiet && skip_quiets {
                continue;
            }
            let hist = if quiet { self.quiet_history(pos, mv, ply, threats) } else { 0 };
            let mut checking_child = None;

            // Search every check evasion: history and exchange estimates
            // cannot justify skipping a forced defensive reply.
            if !root && !in_check && best_score > -MATE_BOUND {
                if quiet {
                    if quiets_tried.len() >= lmp_limit {
                        skip_quiets = true;
                        continue;
                    }
                    let lmr_depth = (depth - self.lmr_reduction(depth, moves_searched + 1)).max(0);
                    if lmr_depth <= FP_MAX_DEPTH.get() && static_eval + FP_BASE.get() + FP_PER_DEPTH.get() * lmr_depth <= alpha {
                        skip_quiets = true;
                        continue;
                    }
                    if lmr_depth <= HIST_PRUNE_MAX_DEPTH && hist < HIST_PRUNE_PER_DEPTH.get() * depth {
                        continue;
                    }
                    if !see_ge(pos, mv, SEE_QUIET_PER_DEPTH.get() * lmr_depth) {
                        continue;
                    }
                } else if depth <= SEE_PRUNE_MAX_DEPTH && !see_ge(pos, mv, SEE_NOISY_PER_DEPTH.get() * depth) {
                    // Material exchanges do not describe forced checking
                    // continuations, especially when recapturers are pinned.
                    let child = pos.make_move(mv);
                    if !child.in_check() {
                        continue;
                    }
                    checking_child = Some(child);
                }
            }

            // Singular extension: if every other move fails well below the
            // TT score, the TT move is forced and deserves more depth.
            let mut extension = 0;
            if !root
                && !singular_search
                && depth >= SE_MIN_DEPTH.get()
                && mv == tt_move
                && let Some(hit) = tt_hit
                && hit.depth >= depth - 3
                && hit.bound != Bound::Upper
                && hit.score.abs() < MATE_BOUND
            {
                let s_beta = hit.score - 2 * depth;
                self.frames[f].excluded = mv;
                let s = self.negamax(pos, s_beta - 1, s_beta, (depth - 1) / 2, ply, cut_node);
                self.frames[f].excluded = Move::NULL;
                if self.aborted() {
                    return 0;
                }
                if s < s_beta {
                    let double = !pv_node && s < s_beta - SE_DOUBLE_MARGIN.get() && (ply as i32) < 2 * self.root_depth;
                    extension = if double { 2 } else { 1 };
                } else if s_beta >= beta {
                    return s_beta; // multi-cut: several moves beat beta
                } else if hit.score >= beta {
                    extension = -1;
                }
            }

            self.frames[f].key = piece_to_key(pos, mv);
            let child = if let Some(child) = checking_child {
                self.update_child_accumulator(pos, mv, &child, ply);
                child
            } else {
                self.make_move(pos, mv, ply)
            };
            if extension == 0 && child.in_check() && see_ge(pos, mv, 0) {
                extension = 1;
            }
            self.hashes.push(child.hash());
            moves_searched += 1;
            let new_depth = depth - 1 + extension;
            let nodes_before = self.nodes;

            let score = if moves_searched == 1 {
                -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, !pv_node && !cut_node)
            } else {
                // Late move reductions, then zero-window PVS with re-searches.
                let mut r = 0;
                if depth >= LMR_MIN_DEPTH && (quiet || order_score < 0) {
                    r = self.lmr_reduction(depth, moves_searched);
                    r += !pv_node as i32;
                    r -= tt_pv as i32;
                    r += !improving as i32;
                    r += cut_node as i32;
                    r += (quiet && tt_capture) as i32;
                    r += (self.frames[f + 1].cutoffs > 3) as i32;
                    r -= child.in_check() as i32;
                    r -= (order_score >= KILLER_1) as i32;
                    if quiet {
                        r -= hist / LMR_HIST_DIVISOR.get();
                    }
                    r = r.min(new_depth - 1).max(0);
                }
                let mut s = -self.negamax(&child, -alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if s > alpha && r > 0 {
                    let adjusted = new_depth + (s > alpha + LMR_DEEPER_MARGIN) as i32 - (s < alpha + LMR_SHALLOWER_MARGIN) as i32;
                    s = -self.negamax(&child, -alpha - 1, -alpha, adjusted, ply + 1, !cut_node);
                }
                if pv_node && s > alpha && s < beta {
                    s = -self.negamax(&child, -beta, -alpha, new_depth, ply + 1, false);
                }
                s
            };
            self.hashes.pop();
            if self.aborted() {
                return 0;
            }
            if root {
                self.root_nodes[history_index(mv)] += self.nodes - nodes_before;
            }
            if quiet {
                quiets_tried.push(mv);
            } else {
                noisies_tried.push(mv);
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mv;
                    if pv_node {
                        self.update_pv(ply, mv);
                    }
                    if score >= beta {
                        self.frames[f].cutoffs += 1;
                        self.update_histories(pos, mv, quiet, &quiets_tried, &noisies_tried, depth, ply, threats);
                        break;
                    }
                }
            }
        }

        if moves_searched == 0 {
            // Only the excluded move was legal (singular verification search).
            return alpha;
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if alpha > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };

        if !in_check
            && !singular_search
            && (best_move.is_null() || !is_noisy(best_move))
            && !(bound == Bound::Lower && best_score <= static_eval)
            && !(bound == Bound::Upper && best_score >= static_eval)
        {
            self.update_correction(pos, raw_eval, best_score, depth, ply);
        }

        if !singular_search {
            self.tt.store(pos.hash(), best_move, best_score, raw_eval, depth, bound, ply, tt_pv);
        }
        best_score
    }

    fn qsearch(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.pv_len[ply] = 0;
        self.nodes += 1;
        self.check_limits();
        if self.aborted() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply + 1);

        let in_check = pos.in_check();
        if self.is_draw(pos) {
            return DRAW;
        }
        if !in_check && !has_legal_move(pos) {
            return DRAW;
        }
        if ply >= MAX_PLY - 1 {
            return if in_check { DRAW } else { self.raw_eval(pos, ply) };
        }

        let pv_node = beta - alpha > 1;
        let tt_hit = self.tt.probe(pos.hash(), ply);
        let tt_move = tt_hit.map_or(Move::NULL, |h| h.mv);
        if let Some(hit) = tt_hit
            && !pv_node
            && pos.halfmove_clock() < 90
            && tt_cutoff(&hit, alpha, beta)
        {
            return hit.score;
        }

        // Stand pat: the side to move may decline every capture. Not
        // available in check, where every evasion is searched instead.
        let (raw_eval, static_eval) = if in_check {
            (NO_EVAL, NO_EVAL)
        } else {
            let raw = match tt_hit {
                Some(hit) if hit.eval != NO_EVAL => hit.eval,
                _ => self.raw_eval(pos, ply),
            };
            (raw, self.corrected_eval(pos, raw, ply))
        };
        let mut best_score = static_eval;
        if !in_check {
            if let Some(hit) = tt_hit
                && pos.halfmove_clock() < 90
                && hit.score.abs() < MATE_BOUND
                && tt_bound_tightens(&hit, static_eval)
            {
                best_score = hit.score;
            }
            if best_score >= beta {
                return best_score;
            }
            alpha = alpha.max(best_score);
        }

        let original_alpha = alpha;
        let mut best_move = Move::NULL;
        let mut moves = MoveList::new();
        if in_check {
            generate_all(pos, &mut moves);
        } else {
            generate_noisy(pos, &mut moves);
        }
        let threats = if in_check { pos.threats() } else { Bitboard::EMPTY };
        let scores = self.score_moves(pos, &moves, tt_move, ply, threats);
        let mut picker = MovePicker::new(moves, scores);
        let futility_base = static_eval + QS_FUTILITY_MARGIN.get();

        while let Some((mv, _)) = picker.next() {
            let mut checking_child = None;
            if !in_check {
                let losing_exchange = !see_ge(pos, mv, 0);
                let gain_bound = futility_base + noisy_gain(pos, mv);
                let futile = !mv.is_promotion() && gain_bound <= alpha;
                if losing_exchange || futile {
                    let child = pos.make_move(mv);
                    if !child.in_check() {
                        if !losing_exchange {
                            best_score = best_score.max(gain_bound);
                        }
                        continue;
                    }
                    checking_child = Some(child);
                }
            }
            self.frames[ply + STACK_OFFSET].key = piece_to_key(pos, mv);
            let child = if let Some(child) = checking_child {
                self.update_child_accumulator(pos, mv, &child, ply);
                child
            } else {
                self.make_move(pos, mv, ply)
            };
            self.hashes.push(child.hash());
            let score = -self.qsearch(&child, -beta, -alpha, ply + 1);
            self.hashes.pop();
            if self.aborted() {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mv;
                    if pv_node {
                        self.update_pv(ply, mv);
                    }
                    if score >= beta {
                        break;
                    }
                }
            }
        }

        if in_check && best_score == -INF {
            return -MATE + ply as i32;
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if alpha > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(pos.hash(), best_move, best_score, raw_eval, 0, bound, ply, pv_node || tt_hit.is_some_and(|h| h.pv));
        best_score
    }
}

#[inline(always)]
fn tt_cutoff(hit: &TTHit, alpha: i32, beta: i32) -> bool {
    match hit.bound {
        Bound::Exact => true,
        Bound::Lower => hit.score >= beta,
        Bound::Upper => hit.score <= alpha,
        Bound::None => false,
    }
}

/// Whether the TT score is a better estimate than `eval` given its bound.
#[inline(always)]
fn tt_bound_tightens(hit: &TTHit, eval: i32) -> bool {
    match hit.bound {
        Bound::Exact => true,
        Bound::Lower => hit.score > eval,
        Bound::Upper => hit.score < eval,
        Bound::None => false,
    }
}

/// Butterfly table for `mv`: side to move, and whether its origin and
/// destination squares are attacked by the opponent.
#[inline(always)]
fn butterfly_table(pos: &Position, mv: Move, threats: Bitboard) -> usize {
    pos.side_to_move().idx() * 4 + (threats.contains(mv.from()) as usize) * 2 + threats.contains(mv.to()) as usize
}

#[cfg(test)]
mod pruning_tests {
    use super::*;

    #[test]
    fn quiescence_searches_losing_exchange_mating_capture() {
        // Both apparent rook recaptures are pinned to Kh8. Material-only
        // exchange estimates miss that Qxg7 is checkmate.
        let pos = Position::from_fen("R5rk/4Q1pr/8/5K2/8/2B5/8/7R w - - 0 1").unwrap();
        assert!(!pos.in_check());
        let mv = legal_moves(&pos).iter().copied()
            .find(|mv| mv.to_uci(false) == "e7g7").unwrap();
        let child = pos.make_move(mv);
        assert!(child.in_check());
        assert!(legal_moves(&child).is_empty());
        assert!(!see_ge(&pos, mv, 0));
        for phase in 0..3 {
            let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
            worker.net = None;
            worker.hashes = vec![pos.hash()];
            let score = match phase {
                0 => worker.qsearch(&pos, -INF, INF, 1),
                1 => worker.qsearch(&pos, MATE - 10, INF, 1),
                _ => worker.negamax(&pos, -INF, INF, 1, 1, false),
            };
            assert_eq!(score, MATE - 2, "search phase {phase}");
            assert_eq!(worker.hashes, vec![pos.hash()]);
        }
    }

    #[test]
    fn checked_node_searches_unfavoured_evasions() {
        // ...Kf8/Kf7 leave Rd7 hanging to Qxd7. ...Kd8 or ...Re7
        // preserve it. Bad history must not remove these legal evasions.
        let pos = Position::from_fen("4k3/3r4/8/8/8/8/8/K2QR3 b - - 0 1").unwrap();
        assert!(pos.in_check());
        let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
        worker.net = None;
        worker.hashes = vec![pos.hash()];
        let threats = pos.threats();
        for &mv in legal_moves(&pos).iter() {
            let bad = matches!(mv.to_uci(false).as_str(), "e8f8" | "e8f7");
            worker.history[butterfly_table(&pos, mv, threats)][history_index(mv)] =
                if bad { HIST_MAX as i16 } else { -HIST_MAX as i16 };
        }
        assert_eq!(worker.negamax(&pos, -INF, INF, 1, 1, false), -900);
    }
}

#[cfg(test)]
mod draw_tests {
    use super::*;

    #[test]
    fn main_search_respects_imminent_fifty_move_draw_before_static_cutoffs() {
        let pos = Position::from_fen("7k/8/8/8/8/2K5/Q7/8 w - - 99 1").unwrap();
        // Every legal move draws at clock100; there is no capture, pawn move
        // or immediate mate. Static evaluation and a virtual pass cannot
        // establish a winning bound here.
        for (depth, alpha, beta) in [(1, 9, 10), (3, 799, 800)] {
            let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
            worker.net = None;
            worker.hashes = vec![pos.hash()];
            worker.frames[STACK_OFFSET].key = 0;
            assert_eq!(worker.negamax(&pos, alpha, beta, depth, 1, false), DRAW,
                       "clock-sensitive pruning at depth{depth}");
            assert_eq!(worker.hashes, vec![pos.hash()]);
        }
    }

    #[test]
    fn game_history_needs_three_occurrences_for_repetition() {
        let mut pos = Position::from_fen("7k/8/8/8/8/2K5/Q7/8 w - - 0 1").unwrap();
        let mut hashes = vec![pos.hash()];
        for cycle in 1..=2 {
            for uci in ["c3b3", "h8g8", "b3c3", "g8h8"] {
                let mv = legal_moves(&pos).iter().copied()
                    .find(|mv| mv.to_uci(false) == uci).unwrap();
                pos = pos.make_move(mv);
                hashes.push(pos.hash());
            }
            let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
            worker.net = None;
            worker.silent = true;
            worker.search(&pos, &hashes, Limits { depth: Some(1), ..Limits::default() });
            assert_eq!(worker.is_draw(&pos), cycle == 2);
        }
    }

    #[test]
    fn stalemate_is_draw_before_stand_pat_or_futility_cutoff() {
        for fen in ["7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
                    "8/8/8/8/8/6q1/5k2/7K w - - 0 1"] {
            let pos = Position::from_fen(fen).unwrap();
            assert!(!pos.in_check());
            assert!(legal_moves(&pos).is_empty());
            let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
            worker.net = None;
            worker.hashes = vec![pos.hash()];
            assert_eq!(worker.qsearch(&pos, -INF, INF, 1), DRAW, "wide-window quiescence: {fen}");
            assert_eq!(worker.qsearch(&pos, -1001, -1000, 1), DRAW, "stand-pat cutoff: {fen}");
            assert_eq!(worker.negamax(&pos, -1001, -1000, 1, 1, false), DRAW, "futility cutoff: {fen}");
        }
    }

    #[test]
    fn quiescence_observes_repetition_and_balances_history() {
        let mut pos = Position::from_fen("7k/8/8/8/8/2K5/Q7/8 w - - 0 1").unwrap();
        let mut hashes = vec![pos.hash()];
        for uci in ["c3b3", "h8g8", "b3c3", "g8h8"] {
            let mv = legal_moves(&pos).iter().copied()
                .find(|mv| mv.to_uci(false) == uci).unwrap();
            pos = pos.make_move(mv);
            hashes.push(pos.hash());
        }
        let mut worker = Worker::new(Arc::new(TT::new(1)), Arc::new(AtomicBool::new(false)));
        worker.net = None;
        worker.hashes = hashes.clone();
        assert_eq!(worker.qsearch(&pos, -INF, INF, 1), DRAW);
        assert_eq!(worker.hashes, hashes);

        let pos = Position::from_fen("4k3/8/3p4/8/8/8/8/3RK3 w - - 0 1").unwrap();
        worker.hashes = vec![pos.hash()];
        assert!(worker.qsearch(&pos, -INF, INF, 1) > DRAW);
        assert_eq!(worker.hashes, vec![pos.hash()]);
    }

    #[test]
    fn quiescence_does_not_reuse_clock_sensitive_tt_cutoffs() {
        let early = Position::from_fen("7k/8/8/8/8/2K5/Q7/8 w - - 0 1").unwrap();
        let late = Position::from_fen("7k/8/8/8/8/2K5/Q7/8 w - - 99 1").unwrap();
        assert_eq!(early.hash(), late.hash());
        let tt = Arc::new(TT::new(1));
        tt.store(early.hash(), Move::NULL, 1500, 900, 0, Bound::Exact, 1, false);
        let mut worker = Worker::new(tt, Arc::new(AtomicBool::new(false)));
        worker.net = None;
        worker.hashes = vec![late.hash()];
        // Narrow window would return the unrelated exact bound without
        // the same clock guard that the main search already applies.
        assert_eq!(worker.qsearch(&late, 899, 900, 1), 900);
    }
}
