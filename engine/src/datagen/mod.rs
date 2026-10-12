//! Self-play data generation and conversion to training records.
//!
//! Each thread plays games from randomised openings at a fixed soft node
//! budget and appends them to a `.drg` game file. `expand` later replays the
//! games and writes filtered 32-byte training positions (`.drp`).

pub mod format;

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::movegen::legal_moves;
use crate::moves::Move;
use crate::position::Position;
use crate::search::tt::TT;
use crate::search::{Limits, MATE_BOUND, Worker};
use crate::types::Color;
use format::{GameRecord, GameResult, UNSCORED};

#[derive(Clone, Debug)]
pub struct DatagenConfig {
    pub threads: usize,
    pub games: u64,
    pub soft_nodes: u64,
    /// Uniformly random plies played from the start position (plus 0 or 1).
    pub random_plies: usize,
    /// Reject openings the engine already scores beyond this (centipawns).
    pub max_opening_score: i32,
    /// Adjudicate a win once |score| stays at or above this…
    pub win_score: i32,
    /// …for this many consecutive plies.
    pub win_plies: u32,
    pub max_plies: usize,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        DatagenConfig {
            threads: 1,
            games: 1000,
            soft_nodes: 5000,
            random_plies: 8,
            max_opening_score: 300,
            win_score: 1000,
            win_plies: 8,
            max_plies: 600,
        }
    }
}

/// xorshift64* — small, fast and good enough for opening randomisation.
struct Rng(u64);

impl Rng {
    fn seeded(stream: u64) -> Rng {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        Rng((nanos ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 32) * n as u64 >> 32) as usize
    }
}

struct Player {
    worker: Worker,
    tt: Arc<TT>,
    stop: Arc<AtomicBool>,
}

impl Player {
    fn new() -> Player {
        let tt = Arc::new(TT::new(16));
        let stop = Arc::new(AtomicBool::new(false));
        let mut worker = Worker::new(tt.clone(), stop.clone());
        worker.silent = true;
        Player { worker, tt, stop }
    }

    fn new_game(&mut self) {
        self.tt.clear();
        self.worker.clear();
    }

    fn search(&mut self, pos: &Position, hashes: &[u64], soft_nodes: u64) -> (Move, i32) {
        self.stop.store(false, Relaxed);
        let limits = Limits { soft_nodes: Some(soft_nodes), nodes: Some(soft_nodes * 20), ..Limits::default() };
        let r = self.worker.search(pos, hashes, limits);
        (r.best_move, r.score)
    }
}

fn random_opening(cfg: &DatagenConfig, rng: &mut Rng, player: &mut Player) -> (Position, Vec<u64>, Vec<(Move, i16)>) {
    loop {
        let mut pos = Position::startpos();
        let mut hashes = vec![pos.hash()];
        let mut plies = Vec::new();
        let count = cfg.random_plies + rng.below(2);
        let mut ok = true;
        for _ in 0..count {
            let moves = legal_moves(&pos);
            if moves.is_empty() {
                ok = false;
                break;
            }
            let mv = moves[rng.below(moves.len())];
            pos = pos.make_move(mv);
            hashes.push(pos.hash());
            plies.push((mv, UNSCORED));
        }
        if !ok || legal_moves(&pos).is_empty() {
            continue;
        }
        player.new_game();
        let (_, score) = player.search(&pos, &hashes, cfg.soft_nodes);
        if score.abs() <= cfg.max_opening_score {
            return (pos, hashes, plies);
        }
    }
}

fn play_game(cfg: &DatagenConfig, rng: &mut Rng, player: &mut Player) -> GameRecord {
    let start_fen = Position::startpos().to_fen();
    let (mut pos, mut hashes, mut plies) = random_opening(cfg, rng, player);
    player.new_game();

    let mut winning_streak = 0u32;
    let mut last_winner = None;
    let result = loop {
        if legal_moves(&pos).is_empty() {
            break if pos.in_check() { GameResult::win_for(!pos.side_to_move()) } else { GameResult::Draw };
        }
        let repetitions = hashes.iter().filter(|&&h| h == pos.hash()).count();
        if pos.halfmove_clock() >= 100 || pos.is_insufficient_material() || repetitions >= 3 || plies.len() >= cfg.max_plies {
            break GameResult::Draw;
        }

        let (mv, score) = player.search(&pos, &hashes, cfg.soft_nodes);
        let stm = pos.side_to_move();
        plies.push((mv, score.clamp(-32000, 32000) as i16));

        if score.abs() > MATE_BOUND {
            // A forced mate was found; trust it rather than playing it out.
            break GameResult::win_for(if score > 0 { stm } else { !stm });
        }
        let white_score = if stm == Color::White { score } else { -score };
        let leader = (white_score.abs() >= cfg.win_score).then_some(white_score > 0);
        if leader.is_some() && leader == last_winner {
            winning_streak += 1;
        } else {
            winning_streak = u32::from(leader.is_some());
        }
        last_winner = leader;
        if winning_streak >= cfg.win_plies {
            break if white_score > 0 { GameResult::WhiteWin } else { GameResult::BlackWin };
        }

        pos = pos.make_move(mv);
        hashes.push(pos.hash());
    };

    GameRecord { start_fen, result, plies }
}

/// Plays `cfg.games` games across `cfg.threads` threads, appending to `out`.
pub fn run(cfg: &DatagenConfig, out: &Path) -> io::Result<()> {
    let is_new = !out.exists() || std::fs::metadata(out)?.len() == 0;
    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    if is_new {
        format::write_game_header(&mut file)?;
    }
    let writer = Mutex::new(BufWriter::with_capacity(1 << 20, file));

    let started = AtomicU64::new(0);
    let finished = AtomicU64::new(0);
    let positions = AtomicU64::new(0);
    let results = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let start = Instant::now();

    println!(
        "datagen: {} games, {} threads, {} soft nodes -> {}",
        cfg.games,
        cfg.threads,
        cfg.soft_nodes,
        out.display()
    );

    std::thread::scope(|s| {
        for t in 0..cfg.threads {
            let (writer, started, finished, positions, results) = (&writer, &started, &finished, &positions, &results);
            s.spawn(move || {
                let mut rng = Rng::seeded(t as u64 + 1);
                let mut player = Player::new();
                let mut buf = Vec::with_capacity(4096);
                while started.fetch_add(1, Relaxed) < cfg.games {
                    let game = play_game(cfg, &mut rng, &mut player);
                    buf.clear();
                    game.encode(&mut buf);
                    writer.lock().unwrap().write_all(&buf).expect("write game");
                    let scored = game.plies.iter().filter(|p| p.1 != UNSCORED).count();
                    positions.fetch_add(scored as u64, Relaxed);
                    results[game.result as usize].fetch_add(1, Relaxed);
                    finished.fetch_add(1, Relaxed);
                }
            });
        }

        let mut last_report = Instant::now();
        while finished.load(Relaxed) < cfg.games {
            std::thread::sleep(Duration::from_millis(200));
            if last_report.elapsed() >= Duration::from_secs(10) {
                last_report = Instant::now();
                report(start, &finished, &positions, &results);
            }
        }
    });

    writer.lock().unwrap().flush()?;
    report(start, &finished, &positions, &results);
    Ok(())
}

fn report(start: Instant, finished: &AtomicU64, positions: &AtomicU64, results: &[AtomicU64; 3]) {
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    let pos = positions.load(Relaxed);
    println!(
        "games {} | positions {} ({:.0}/s) | white {} draw {} black {} | {:.0}s",
        finished.load(Relaxed),
        pos,
        pos as f64 / secs,
        results[2].load(Relaxed),
        results[1].load(Relaxed),
        results[0].load(Relaxed),
        secs
    );
}

/// Replays games from `input` and writes filtered training positions.
/// Skipped: unscored plies, positions in check, positions whose chosen move
/// is a capture or promotion (score is not quiet), and mate scores.
pub fn expand(input: &Path, output: &Path) -> io::Result<(u64, u64)> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(input)?);
    format::read_game_header(&mut reader)?;
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(output)?);

    let (mut games, mut written) = (0u64, 0u64);
    while let Some(game) = GameRecord::decode(&mut reader)? {
        let mut pos = Position::from_fen(&game.start_fen).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for &(raw, score) in &game.plies {
            // Validate against the legal move list; never trust stored bits.
            let Some(mv) = legal_moves(&pos).iter().copied().find(|&m| m == raw) else {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("illegal move in game {games}")));
            };
            let tactical = mv.is_capture() || mv.is_promotion();
            let keep = score != UNSCORED && !pos.in_check() && !tactical && (score as i32).abs() < MATE_BOUND;
            if keep {
                let white_score = if pos.side_to_move() == Color::White { score } else { -score };
                writer.write_all(&format::pack_position(&pos, white_score, game.result))?;
                written += 1;
            }
            pos = pos.make_move(mv);
        }
        games += 1;
    }
    writer.flush()?;
    Ok((games, written))
}
