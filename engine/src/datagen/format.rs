//! DarkReality data formats.
//!
//! **Game file (`.drg`)**: what datagen writes. Compact (≈4 bytes per ply), so
//! billions of positions fit on a small disk and upload quickly.
//!
//! ```text
//! header  "DRGAMES\0" u32 version
//! game    u8 fen_len, fen bytes, u8 result (0 black win, 1 draw, 2 white win),
//!         u16 ply_count, then per ply: u16 move (engine encoding),
//!         i16 score (side-to-move POV; i16::MIN = not scored)
//! ```
//!
//! **Position file (`.drp`)**: what the trainer reads, made by `expand`.
//! Fixed 32-byte records, all values little-endian:
//!
//! ```text
//! u64 occupancy            (a1 = bit 0)
//! [u8; 16] pieces          4-bit piece codes ((type << 1) | color) for each
//!                          occupied square in ascending order, low nibble first
//! i16 score                white POV, centipawns
//! u8  result               0 black win, 1 draw, 2 white win
//! u8  side to move         0 white, 1 black
//! u16 fullmove number
//! [u8; 2] flags            byte 0 bit 0: no observed game result (score only);
//!                          all zero for our self-play records
//! ```

use std::io::{self, Read, Write};

use crate::moves::Move;
use crate::position::Position;
use crate::types::Color;

pub const GAME_MAGIC: &[u8; 8] = b"DRGAMES\0";
pub const GAME_VERSION: u32 = 1;
pub const UNSCORED: i16 = i16::MIN;
pub const RECORD_SIZE: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    BlackWin = 0,
    Draw = 1,
    WhiteWin = 2,
}

impl GameResult {
    pub fn win_for(color: Color) -> GameResult {
        match color {
            Color::White => GameResult::WhiteWin,
            Color::Black => GameResult::BlackWin,
        }
    }

    fn from_u8(v: u8) -> Option<GameResult> {
        match v {
            0 => Some(GameResult::BlackWin),
            1 => Some(GameResult::Draw),
            2 => Some(GameResult::WhiteWin),
            _ => None,
        }
    }
}

pub struct GameRecord {
    pub start_fen: String,
    pub result: GameResult,
    /// Each move played and the mover's search score (or `UNSCORED`).
    pub plies: Vec<(Move, i16)>,
}

impl GameRecord {
    pub fn encode(&self, out: &mut Vec<u8>) {
        let fen = self.start_fen.as_bytes();
        out.push(fen.len() as u8);
        out.extend_from_slice(fen);
        out.push(self.result as u8);
        out.extend_from_slice(&(self.plies.len() as u16).to_le_bytes());
        for &(mv, score) in &self.plies {
            out.extend_from_slice(&mv.raw().to_le_bytes());
            out.extend_from_slice(&score.to_le_bytes());
        }
    }

    /// Reads the next game; `Ok(None)` at a clean end of file. A truncated
    /// final game (e.g. datagen killed mid-write) also yields `Ok(None)`.
    pub fn decode(r: &mut impl Read) -> io::Result<Option<GameRecord>> {
        let mut len = [0u8; 1];
        if r.read(&mut len)? == 0 {
            return Ok(None);
        }
        let mut fen = vec![0u8; len[0] as usize];
        let mut head = [0u8; 3];
        if read_all(r, &mut fen)? || read_all(r, &mut head)? {
            return Ok(None);
        }
        let result = GameResult::from_u8(head[0]).ok_or_else(|| bad_data("bad game result"))?;
        let count = u16::from_le_bytes([head[1], head[2]]) as usize;
        let mut body = vec![0u8; count * 4];
        if read_all(r, &mut body)? {
            return Ok(None);
        }
        let plies = body
            .chunks_exact(4)
            .map(|c| {
                let mv = Move::from_raw(u16::from_le_bytes([c[0], c[1]]));
                (mv, i16::from_le_bytes([c[2], c[3]]))
            })
            .collect();
        let start_fen = String::from_utf8(fen).map_err(|_| bad_data("FEN is not UTF-8"))?;
        Ok(Some(GameRecord { start_fen, result, plies }))
    }
}

/// Fills `buf`; returns `true` if the stream ended before it was full.
fn read_all(r: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    match r.read_exact(buf) {
        Ok(()) => Ok(false),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(true),
        Err(e) => Err(e),
    }
}

fn bad_data(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn write_game_header(w: &mut impl Write) -> io::Result<()> {
    w.write_all(GAME_MAGIC)?;
    w.write_all(&GAME_VERSION.to_le_bytes())
}

pub fn read_game_header(r: &mut impl Read) -> io::Result<()> {
    let mut header = [0u8; 12];
    r.read_exact(&mut header)?;
    if &header[..8] != GAME_MAGIC {
        return Err(bad_data("not a DarkReality game file"));
    }
    let version = u32::from_le_bytes(header[8..].try_into().unwrap());
    if version != GAME_VERSION {
        return Err(bad_data("unsupported game file version"));
    }
    Ok(())
}

/// Packs a training position (see module docs).
pub fn pack_position(pos: &Position, white_score: i16, result: GameResult) -> [u8; RECORD_SIZE] {
    let mut rec = [0u8; RECORD_SIZE];
    let occ = pos.occupied();
    rec[0..8].copy_from_slice(&occ.0.to_le_bytes());
    for (i, sq) in occ.enumerate() {
        let code = pos.piece_on(sq).idx() as u8;
        rec[8 + i / 2] |= code << (4 * (i % 2));
    }
    rec[24..26].copy_from_slice(&white_score.to_le_bytes());
    rec[26] = result as u8;
    rec[27] = pos.side_to_move() as u8;
    rec[28..30].copy_from_slice(&pos.fullmove_number().to_le_bytes());
    rec
}
