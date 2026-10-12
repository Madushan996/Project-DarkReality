use std::path::Path;
use std::process::exit;

use darkreality::datagen::{self, DatagenConfig};
use darkreality::{bench, nnue, perft, position::Position, uci::Uci};

fn usage() -> ! {
    eprintln!(
        "usage:
  darkreality                               UCI mode
  darkreality bench [depth]
  darkreality perft <depth> [fen]
  darkreality datagen --out <file.drg> [--games N] [--threads N] [--nodes N] [--random-plies N]
  darkreality expand <in.drg> <out.drp>"
    );
    exit(2)
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("{msg}");
    exit(1)
}

fn parse<T: std::str::FromStr>(value: Option<&String>, name: &str) -> T {
    value.and_then(|v| v.parse().ok()).unwrap_or_else(|| fail(format!("bad value for {name}")))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH);
            bench::run(depth);
        }
        Some("perft") => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
            let pos = if args.len() > 2 {
                Position::from_fen(&args[2..].join(" ")).unwrap_or_else(|e| fail(e))
            } else {
                Position::startpos()
            };
            perft::divide(&pos, depth, false);
        }
        Some("datagen") => {
            let mut cfg = DatagenConfig::default();
            let mut out = None;
            let mut it = args[1..].iter();
            while let Some(flag) = it.next() {
                match flag.as_str() {
                    "--out" => out = it.next().cloned(),
                    "--games" => cfg.games = parse(it.next(), flag),
                    "--threads" => cfg.threads = parse(it.next(), flag),
                    "--nodes" => cfg.soft_nodes = parse(it.next(), flag),
                    "--random-plies" => cfg.random_plies = parse(it.next(), flag),
                    _ => usage(),
                }
            }
            let out = out.unwrap_or_else(|| usage());
            let eval = if nnue::network().is_some() { "embedded NNUE" } else { "material (Gen0)" };
            println!("evaluation: {eval}");
            datagen::run(&cfg, Path::new(&out)).unwrap_or_else(|e| fail(e));
        }
        Some("expand") => {
            let (Some(input), Some(output)) = (args.get(1), args.get(2)) else { usage() };
            let (games, positions) = datagen::expand(Path::new(input), Path::new(output)).unwrap_or_else(|e| fail(e));
            println!("{games} games -> {positions} positions");
        }
        Some(_) => usage(),
        None => Uci::new().run(),
    }
}
