use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread::JoinHandle;

use crate::movegen::legal_moves;
use crate::moves::Move;
use crate::perft::divide;
use crate::position::Position;
use crate::search::tt::TT;
use crate::search::{Limits, Worker};

pub const NAME: &str = "DarkReality";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const DEFAULT_HASH_MB: usize = 16;
const MAX_HASH_MB: usize = 65536;

pub struct Uci {
    pos: Position,
    /// Hashes of every position since the game start, ending with `pos`.
    game_hashes: Vec<u64>,
    chess960: bool,
    move_overhead: u64,
    tt: Arc<TT>,
    stop: Arc<AtomicBool>,
    /// The idle worker, or `None` while it is lent to the search thread.
    worker: Option<Box<Worker>>,
    search_thread: Option<JoinHandle<Box<Worker>>>,
}

impl Default for Uci {
    fn default() -> Self {
        Uci::new()
    }
}

impl Uci {
    pub fn new() -> Uci {
        let tt = Arc::new(TT::new(DEFAULT_HASH_MB));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = Box::new(Worker::new(tt.clone(), stop.clone()));
        let pos = Position::startpos();
        Uci {
            game_hashes: vec![pos.hash()],
            pos,
            chess960: false,
            move_overhead: 10,
            tt,
            stop,
            worker: Some(worker),
            search_thread: None,
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if !self.handle(line.trim()) {
                break;
            }
        }
        self.stop.store(true, Relaxed);
        self.wait_for_search();
    }

    /// Handles one command; returns `false` on `quit`.
    pub fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => {
                println!("id name {NAME} {VERSION} {}", env!("DR_NET_NAME"));
                println!("id author Madushan Dissanayake");
                println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name Move Overhead type spin default 10 min 0 max 5000");
                println!("option name UCI_Chess960 type check default false");
                for t in crate::search::params::TUNABLES {
                    println!("option name {} type spin default {} min {} max {}", t.name, t.default, t.min, t.max);
                }
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {
                self.wait_for_search();
                self.tt.clear();
                if let Some(w) = self.worker.as_mut() {
                    w.clear();
                }
                self.set_position(&["startpos"]).ok();
            }
            Some("position") => {
                if let Err(e) = self.set_position(&tokens[1..]) {
                    println!("info string {e}");
                }
            }
            Some("go") => self.go(&tokens[1..]),
            Some("stop") => {
                self.stop.store(true, Relaxed);
                self.wait_for_search();
            }
            Some("setoption") => self.set_option(&tokens[1..]),
            Some("d") => println!("{}", self.pos),
            Some("eval") => match crate::static_eval(&self.pos) {
                (cp, Some(units)) => println!("static eval: {cp} cp (side to move), network units {units}"),
                (cp, None) => println!("static eval: {cp} cp (side to move), material"),
            },
            Some("wait") => self.wait_for_search(),
            Some("quit") => return false,
            Some(cmd) => println!("info string unknown command '{cmd}'"),
            None => {}
        }
        let _ = io::stdout().flush();
        true
    }

    /// Blocks until a running search has printed `bestmove`.
    fn wait_for_search(&mut self) {
        if let Some(handle) = self.search_thread.take() {
            self.worker = Some(handle.join().expect("search thread panicked"));
        }
    }

    fn set_position(&mut self, args: &[&str]) -> Result<(), String> {
        let moves_at = args.iter().position(|&t| t == "moves").unwrap_or(args.len());
        let (setup, moves) = args.split_at(moves_at);

        let mut pos = match setup.first() {
            Some(&"startpos") => Position::startpos(),
            Some(&"fen") => Position::from_fen(&setup[1..].join(" ")).map_err(|e| e.to_string())?,
            _ => return Err("expected 'startpos' or 'fen'".to_string()),
        };
        let mut hashes = vec![pos.hash()];
        for text in moves.iter().skip(1) {
            let mv = self.parse_move(&pos, text).ok_or_else(|| format!("illegal move '{text}'"))?;
            pos = pos.make_move(mv);
            hashes.push(pos.hash());
        }
        self.pos = pos;
        self.game_hashes = hashes;
        Ok(())
    }

    fn parse_move(&self, pos: &Position, text: &str) -> Option<Move> {
        legal_moves(pos).iter().copied().find(|mv| mv.to_uci(self.chess960) == text)
    }

    fn go(&mut self, args: &[&str]) {
        if args.first() == Some(&"perft") {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(1);
            divide(&self.pos, depth, self.chess960);
            return;
        }

        self.wait_for_search();
        let limits = parse_limits(args);
        let mut worker = self.worker.take().expect("worker is idle");
        worker.set_tt(self.tt.clone());
        worker.chess960 = self.chess960;
        worker.move_overhead = self.move_overhead;

        let pos = self.pos;
        let hashes = self.game_hashes.clone();
        self.stop.store(false, Relaxed);
        self.search_thread = Some(std::thread::spawn(move || {
            let result = worker.search(&pos, &hashes, limits);
            println!("bestmove {}", result.best_move.to_uci(worker.chess960));
            let _ = io::stdout().flush();
            worker
        }));
    }

    fn set_option(&mut self, args: &[&str]) {
        // setoption name <id...> [value <x...>]
        let value_at = args.iter().position(|&t| t == "value").unwrap_or(args.len());
        let name = args.get(1..value_at).unwrap_or(&[]).join(" ").to_ascii_lowercase();
        let value = args.get(value_at + 1..).unwrap_or(&[]).join(" ");
        self.wait_for_search();
        match name.as_str() {
            "hash" => match value.parse::<usize>() {
                Ok(mb) => self.tt = Arc::new(TT::new(mb.clamp(1, MAX_HASH_MB))),
                Err(_) => println!("info string bad Hash value '{value}'"),
            },
            "threads" => {}
            "move overhead" => self.move_overhead = value.parse().unwrap_or(self.move_overhead),
            "uci_chess960" => self.chess960 = value.eq_ignore_ascii_case("true"),
            _ => match (crate::search::params::find_tunable(&name), value.parse::<i32>()) {
                (Some(t), Ok(v)) => t.set(v),
                _ => println!("info string unknown option '{name}'"),
            },
        }
    }
}

fn parse_limits(args: &[&str]) -> Limits {
    fn num<T: FromStr>(v: Option<&&str>) -> Option<T> {
        v.and_then(|s| s.parse().ok())
    }
    let mut limits = Limits::default();
    let mut it = args.iter();
    while let Some(&tok) = it.next() {
        match tok {
            "depth" => limits.depth = num(it.next()),
            "nodes" => limits.nodes = num(it.next()),
            "movetime" => limits.movetime = num(it.next()),
            "wtime" => limits.time[0] = num::<i64>(it.next()).map(|t| t.max(0) as u64),
            "btime" => limits.time[1] = num::<i64>(it.next()).map(|t| t.max(0) as u64),
            "winc" => limits.inc[0] = num(it.next()).unwrap_or(0),
            "binc" => limits.inc[1] = num(it.next()).unwrap_or(0),
            "movestogo" => limits.movestogo = num(it.next()),
            "infinite" => limits.infinite = true,
            _ => {}
        }
    }
    let bounded = limits.depth.is_some()
        || limits.nodes.is_some()
        || limits.movetime.is_some()
        || limits.time.iter().any(Option::is_some);
    if !bounded {
        limits.infinite = true;
    }
    limits
}
