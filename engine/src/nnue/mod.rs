//! NNUE evaluation: (768 → HIDDEN)×2 perspectives → OUTPUT_BUCKETS, SCReLU.
//! The output bucket is chosen by the number of pieces on the board.
//!
//! Network file (`.nnue`, little-endian), written by `trainer/train.py`:
//!
//! ```text
//! v3: "DRNNUE03"  u32 inputs  u32 hidden  u32 buckets  u32 king_buckets  u32 qa  u32 qb  u32 scale
//! v2: "DRNNUE02"  u32 inputs  u32 hidden  u32 buckets  u32 qa  u32 qb  u32 scale
//! v1: "DRNNUE01"  u32 inputs  u32 hidden  u32 qa  u32 qb  u32 scale   (buckets = 1)
//! i16 ft_weights[inputs][hidden]   i16 ft_bias[hidden]
//! i16 out_weights[buckets][2][hidden]   (side to move first)
//! i32 out_bias[buckets]
//! ```
//!
//! Feature index for perspective `p` (inputs = 768 x king_buckets):
//! `king_bucket * 768 + (piece colour != p) * 384 + piece_type * 64 + square'`
//! where `square'` is flipped vertically for black. Four contexts also
//! mirror horizontally whenever p's own king stands on files e-h. Sixty-four
//! contexts use the perspective king square directly, without horizontal
//! mirroring. With one context this is exactly the v1/v2 layout.

use std::sync::OnceLock;

use crate::moves::Move;
use crate::bitboard::Bitboard;
use crate::position::Position;
use crate::types::{Color, Piece, PieceType, Square};

// `HIDDEN`, `OUTPUT_BUCKETS`, `KING_BUCKETS` and `NET_VERSION`, read from the
// embedded network's header by build.rs.
include!(concat!(env!("OUT_DIR"), "/net_arch.rs"));
pub const INPUTS: usize = 768 * KING_BUCKETS;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;
const MAGIC: &[u8; 8] = match NET_VERSION {
    1 => b"DRNNUE01",
    2 => b"DRNNUE02",
    _ => b"DRNNUE03",
};

/// King bucket by square in the perspective frame after mirroring (files
/// a-d): back-rank corner, back-rank centre, second rank, everything else.
const KING_BUCKET_MAP: [u8; 64] = {
    let mut map = [3u8; 64];
    let mut sq = 0;
    while sq < 64 {
        map[sq] = match (sq / 8, sq % 8) {
            (0, 0 | 1 | 6 | 7) => 0,
            (0, _) => 1,
            (1, _) => 2,
            _ => 3,
        };
        sq += 1;
    }
    map
};

/// Header words after the magic, in file order.
fn expected_header() -> Vec<u32> {
    let mut h = vec![INPUTS as u32, HIDDEN as u32];
    if NET_VERSION >= 2 {
        h.push(OUTPUT_BUCKETS as u32);
    }
    if NET_VERSION >= 3 {
        h.push(KING_BUCKETS as u32);
    }
    h.extend([QA as u32, QB as u32, SCALE as u32]);
    h
}

#[repr(C, align(64))]
pub struct Network {
    ft_weights: [[i16; HIDDEN]; INPUTS],
    ft_bias: [i16; HIDDEN],
    out_weights: [[[i16; HIDDEN]; 2]; OUTPUT_BUCKETS],
    out_bias: [i32; OUTPUT_BUCKETS],
}

#[cfg(dr_net)]
static NET_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net.nnue"));
#[cfg(not(dr_net))]
static NET_BYTES: &[u8] = &[];

static NETWORK: OnceLock<Option<Box<Network>>> = OnceLock::new();

/// The embedded network, or `None` for a build without one (material eval).
pub fn network() -> Option<&'static Network> {
    NETWORK
        .get_or_init(|| (!NET_BYTES.is_empty()).then(|| Network::parse(NET_BYTES).expect("embedded network is valid")))
        .as_deref()
}

impl Network {
    pub fn parse(bytes: &[u8]) -> Result<Box<Network>, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != MAGIC {
            return Err("not a DarkReality network".into());
        }
        let expected = expected_header();
        let header: Vec<u32> = (0..expected.len()).map(|_| r.u32()).collect::<Result<_, _>>()?;
        if header != expected {
            return Err(format!("network header {header:?}, engine expects {expected:?}"));
        }

        // Zeroed heap allocation: the struct is too large for the stack at
        // bigger sizes, and all-zero bits are a valid Network.
        let mut net: Box<Network> = unsafe { Box::new_zeroed().assume_init() };
        for row in net.ft_weights.iter_mut() {
            for w in row.iter_mut() {
                *w = r.i16()?;
            }
        }
        for b in net.ft_bias.iter_mut() {
            *b = r.i16()?;
        }
        for bucket in net.out_weights.iter_mut() {
            for row in bucket.iter_mut() {
                for w in row.iter_mut() {
                    *w = r.i16()?;
                }
            }
        }
        for b in net.out_bias.iter_mut() {
            *b = r.u32()? as i32;
        }
        if r.at != bytes.len() {
            return Err(format!("{} trailing bytes in network", bytes.len() - r.at));
        }
        Ok(net)
    }

    /// Evaluation in centipawns (pawn ≈ 100) from the side to move's view,
    /// for a position with `pieces` pieces on the board (kings included).
    ///
    /// The trainer fits `sigmoid(units / 400)` (natural log), so raw network
    /// units are ln(10) ≈ 2.303 times Elo-style centipawns; search margins
    /// and SEE values are in centipawns, so convert here.
    pub fn evaluate(&self, acc: &Accumulator, stm: Color, pieces: u32) -> i32 {
        (self.evaluate_units(acc, stm, pieces) as i64 * 1000 / 2303) as i32
    }

    /// Raw network output units (what the trainer's integer check reproduces).
    pub fn evaluate_units(&self, acc: &Accumulator, stm: Color, pieces: u32) -> i32 {
        let bucket = output_bucket(pieces);
        let us = &acc.values[stm.idx()];
        let them = &acc.values[(!stm).idx()];
        let w = &self.out_weights[bucket];
        let sum = screlu_dot(us, &w[0]) + screlu_dot(them, &w[1]);
        ((sum / QA as i64 + self.out_bias[bucket] as i64) * SCALE as i64 / (QA * QB) as i64) as i32
    }
}

/// Output bucket for a piece count: equal-width bands over 2..=32 pieces.
#[inline(always)]
pub fn output_bucket(pieces: u32) -> usize {
    const WIDTH: usize = 32_usize.div_ceil(OUTPUT_BUCKETS);
    ((pieces as usize).saturating_sub(2) / WIDTH).min(OUTPUT_BUCKETS - 1)
}

/// Σ clamp(x, 0, QA)² · w over one perspective.
///
/// The trainer keeps |w| ≤ 127, so clamp(x)·w fits in i16 and the AVX2 path
/// can multiply it by clamp(x) again with `madd`: every product and pair sum
/// fits in i32, and each lane accumulates at most HIDDEN/16 pairs.
#[inline(always)]
fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: AVX2 is enabled at compile time; HIDDEN is a multiple of 16
        // and both arrays hold HIDDEN elements, so every load is in bounds.
        unsafe { screlu_dot_avx2(acc, weights) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum = 0i64;
        for i in 0..HIDDEN {
            let x = (acc[i] as i32).clamp(0, QA);
            sum += (x * x * weights[i] as i32) as i64;
        }
        sum
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn screlu_dot_avx2(acc: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i64 {
    use std::arch::x86_64::*;
    const { assert!(HIDDEN % 16 == 0) };
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        for i in (0..HIDDEN).step_by(16) {
            let x = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
            let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(x, w), x));
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
        lanes.iter().map(|&v| v as i64).sum()
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let slice = self.bytes.get(self.at..self.at + n).ok_or("network file is truncated")?;
        self.at += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16, String> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
}

/// Where one perspective's features live, given its own king's square.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Frame {
    base: usize,
    xor: u8,
}

#[inline(always)]
fn frame(perspective: Color, king: Square) -> Frame {
    let flip = if perspective == Color::White { 0 } else { 56 };
    if KING_BUCKETS == 1 {
        return Frame { base: 0, xor: flip };
    }
    let rel = king.raw() ^ flip;
    if KING_BUCKETS == 64 {
        return Frame { base: rel as usize * 768, xor: flip };
    }
    let mirror = if rel & 7 >= 4 { 7 } else { 0 };
    Frame { base: KING_BUCKET_MAP[(rel ^ mirror) as usize] as usize * 768, xor: flip ^ mirror }
}

#[inline(always)]
fn feature(perspective: Color, fr: Frame, piece: Piece, sq: Square) -> usize {
    let relative_color = (piece.color() != perspective) as usize;
    fr.base + relative_color * 384 + piece.piece_type().idx() * 64 + (sq.raw() ^ fr.xor) as usize
}

/// Hidden-layer pre-activations for both perspectives (`[white, black]`).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    values: [[i16; HIDDEN]; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator { values: [[0; HIDDEN]; 2] }
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
struct RefreshEntry {
    values: [i16; HIDDEN],
    pieces: [Bitboard; 12],
    valid: bool,
}

/// Previous boards/accumulators for each perspective's own king square.
/// A revisit changes only the piece-square rows that differ from that board.
pub struct RefreshCache {
    entries: Box<[RefreshEntry]>,
    network_id: usize,
}

impl Default for RefreshCache {
    fn default() -> Self {
        Self {
            entries: vec![RefreshEntry {
                values: [0; HIDDEN], pieces: [Bitboard::EMPTY; 12], valid: false,
            }; if KING_BUCKETS > 1 { 128 } else { 0 }].into_boxed_slice(),
            network_id: 0,
        }
    }
}

impl RefreshCache {
    pub fn clear(&mut self) {
        self.network_id = 0;
        for e in &mut self.entries { e.valid = false; }
    }

    fn refresh_side(&mut self, net: &Network, pos: &Position, p: Color, out: &mut [i16; HIDDEN]) {
        let identity = net as *const Network as usize;
        if self.network_id != identity {
            self.clear();
            self.network_id = identity;
        }
        let index = p.idx() * 64 + pos.king_sq(p).raw() as usize;
        let e = &mut self.entries[index];
        if !e.valid {
            e.values = net.ft_bias;
            e.pieces = [Bitboard::EMPTY; 12];
            e.valid = true;
        }
        let fr = frame(p, pos.king_sq(p));
        for color in Color::ALL {
            for pt in PieceType::ALL {
                let slot = color.idx() * 6 + pt.idx();
                let now = pos.pieces_of(color, pt);
                for sq in e.pieces[slot] & !now {
                    let row = &net.ft_weights[feature(p, fr, Piece::new(color, pt), sq)];
                    for i in 0..HIDDEN { e.values[i] = e.values[i].wrapping_sub(row[i]); }
                }
                for sq in now & !e.pieces[slot] {
                    add(&mut e.values, &net.ft_weights[feature(p, fr, Piece::new(color, pt), sq)]);
                }
                e.pieces[slot] = now;
            }
        }
        *out = e.values;
    }
}

impl Accumulator {
    pub fn refresh(&mut self, net: &Network, pos: &Position) {
        for p in Color::ALL {
            self.refresh_side(net, pos, p);
        }
    }

    fn refresh_side(&mut self, net: &Network, pos: &Position, p: Color) {
        let fr = frame(p, pos.king_sq(p));
        let out = &mut self.values[p.idx()];
        *out = net.ft_bias;
        for sq in pos.occupied() {
            add(out, &net.ft_weights[feature(p, fr, pos.piece_on(sq), sq)]);
        }
    }

    /// Sets `self` to `parent` updated for `mv` played in `pos` (the parent
    /// position), with `child` the position after the move. Only changed
    /// features are touched, except that a perspective whose own king moved
    /// to a different bucket or board half is rebuilt from `child`.
    pub fn update_from(&mut self, parent: &Accumulator, net: &Network, pos: &Position, mv: Move, child: &Position) {
        self.update_impl(parent, net, pos, mv, child, None);
    }

    pub fn update_from_cached(&mut self, parent: &Accumulator, net: &Network, pos: &Position, mv: Move, child: &Position, cache: &mut RefreshCache) {
        self.update_impl(parent, net, pos, mv, child, Some(cache));
    }

    fn update_impl(&mut self, parent: &Accumulator, net: &Network, pos: &Position, mv: Move, child: &Position, mut cache: Option<&mut RefreshCache>) {
        let us = pos.side_to_move();
        let from = mv.from();
        let to = mv.to();
        let moved = pos.piece_on(from);

        // At most 2 additions and 2 removals (castling, promotion-captures).
        let mut adds: [(Piece, Square); 2] = [(Piece::NONE, from); 2];
        let mut subs: [(Piece, Square); 2] = [(Piece::NONE, from); 2];
        let (n_add, n_sub);

        if mv.is_castle() {
            let king_dest = mv.castle_king_dest();
            let rook_dest = Square::from_coords(if to.file() > from.file() { 5 } else { 3 }, from.rank());
            let rook = pos.piece_on(to);
            subs = [(moved, from), (rook, to)];
            adds = [(moved, king_dest), (rook, rook_dest)];
            (n_add, n_sub) = (2, 2);
        } else {
            let placed = mv.promo_piece().map_or(moved, |pt| Piece::new(us, pt));
            subs[0] = (moved, from);
            adds[0] = (placed, to);
            n_add = 1;
            n_sub = if mv.is_capture() {
                let cap_sq = if mv.is_en_passant() { Square::new(to.raw() ^ 8) } else { to };
                subs[1] = (pos.piece_on(cap_sq), cap_sq);
                2
            } else {
                1
            };
        }

        // One fused pass per perspective (parent ± rows) instead of a copy
        // followed by a pass per changed feature.
        for p in Color::ALL {
            let fr = frame(p, child.king_sq(p));
            let king_moved = moved.piece_type() == PieceType::King && p == us;
            if KING_BUCKETS > 1 && king_moved && fr != frame(p, pos.king_sq(p)) {
                if let Some(c) = &mut cache {
                    c.refresh_side(net, child, p, &mut self.values[p.idx()]);
                } else {
                    self.refresh_side(net, child, p);
                }
                continue;
            }
            let out = &mut self.values[p.idx()];
            let from = &parent.values[p.idx()];
            let row = |(piece, sq): (Piece, Square)| &net.ft_weights[feature(p, fr, piece, sq)];
            match (n_add, n_sub) {
                (1, 1) => fused_1_1(out, from, row(adds[0]), row(subs[0])),
                (1, 2) => fused_1_2(out, from, row(adds[0]), row(subs[0]), row(subs[1])),
                _ => fused_2_2(out, from, row(adds[0]), row(adds[1]), row(subs[0]), row(subs[1])),
            }
        }
    }
}

#[inline(always)]
fn fused_1_1(out: &mut [i16; HIDDEN], from: &[i16; HIDDEN], a: &[i16; HIDDEN], s: &[i16; HIDDEN]) {
    for i in 0..HIDDEN {
        out[i] = from[i].wrapping_add(a[i]).wrapping_sub(s[i]);
    }
}

#[inline(always)]
fn fused_1_2(out: &mut [i16; HIDDEN], from: &[i16; HIDDEN], a: &[i16; HIDDEN], s1: &[i16; HIDDEN], s2: &[i16; HIDDEN]) {
    for i in 0..HIDDEN {
        out[i] = from[i].wrapping_add(a[i]).wrapping_sub(s1[i]).wrapping_sub(s2[i]);
    }
}

#[inline(always)]
fn fused_2_2(
    out: &mut [i16; HIDDEN],
    from: &[i16; HIDDEN],
    a1: &[i16; HIDDEN],
    a2: &[i16; HIDDEN],
    s1: &[i16; HIDDEN],
    s2: &[i16; HIDDEN],
) {
    for i in 0..HIDDEN {
        out[i] = from[i].wrapping_add(a1[i]).wrapping_add(a2[i]).wrapping_sub(s1[i]).wrapping_sub(s2[i]);
    }
}

#[inline(always)]
fn add(acc: &mut [i16; HIDDEN], weights: &[i16; HIDDEN]) {
    for (a, &w) in acc.iter_mut().zip(weights) {
        *a = a.wrapping_add(w);
    }
}

/// Feature index helper exposed for tests and tools.
pub fn feature_index(perspective: Color, king: Square, piece_type: PieceType, piece_color: Color, sq: Square) -> usize {
    feature(perspective, frame(perspective, king), Piece::new(piece_color, piece_type), sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    /// A network with pseudo-random weights, built through the real parser.
    fn random_network() -> Box<Network> {
        let mut bytes = MAGIC.to_vec();
        for v in expected_header() {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        for _ in 0..(INPUTS * HIDDEN + HIDDEN + OUTPUT_BUCKETS * 2 * HIDDEN) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.extend_from_slice(&((state % 201) as i16 - 100).to_le_bytes());
        }
        for _ in 0..OUTPUT_BUCKETS {
            bytes.extend_from_slice(&1234i32.to_le_bytes());
        }
        Network::parse(&bytes).unwrap()
    }

    fn check_tree(net: &Network, pos: &Position, acc: &Accumulator, depth: u32, cache: &mut RefreshCache) {
        let mut fresh = Accumulator::default();
        fresh.refresh(net, pos);
        assert!(fresh.values == acc.values, "incremental accumulator drifted at {}", pos.to_fen());
        if depth == 0 {
            return;
        }
        for &mv in legal_moves(pos).iter() {
            let mut child = Accumulator::default();
            let child_pos = pos.make_move(mv);
            child.update_from_cached(acc, net, pos, mv, &child_pos, cache);
            check_tree(net, &child_pos, &child, depth - 1, cache);
        }
    }

    #[test]
    fn cached_king_refresh_tracks_network_and_explicit_clear() {
        if KING_BUCKETS == 1 {
            return;
        }
        let first = random_network();
        let mut second = random_network();
        for bias in &mut second.ft_bias {
            *bias = bias.wrapping_add(17);
        }
        let pos = Position::from_fen("8/8/2k5/8/2K5/8/3Q4/8 w - - 0 1").unwrap();
        let mv = *legal_moves(&pos).iter().find(|&&mv|
            pos.piece_on(mv.from()).piece_type() == PieceType::King).unwrap();
        let child_pos = pos.make_move(mv);
        let mut cache = RefreshCache::default();
        for net in [first.as_ref(), second.as_ref(), first.as_ref()] {
            let mut parent = Accumulator::default();
            parent.refresh(net, &pos);
            let mut cached = Accumulator::default();
            cached.update_from_cached(&parent, net, &pos, mv, &child_pos, &mut cache);
            let mut independent = Accumulator::default();
            independent.refresh(net, &child_pos);
            assert!(cached.values == independent.values);
        }
        assert!(cache.entries.iter().any(|entry| entry.valid));
        cache.clear();
        assert!(cache.entries.iter().all(|entry| !entry.valid));
        assert_eq!(cache.network_id, 0);
        let mut parent = Accumulator::default();
        parent.refresh(&first, &pos);
        let mut cached = Accumulator::default();
        cached.update_from_cached(&parent, &first, &pos, mv, &child_pos, &mut cache);
        let mut independent = Accumulator::default();
        independent.refresh(&first, &child_pos);
        assert!(cached.values == independent.values);
    }

    #[test]
    fn incremental_updates_match_refresh() {
        let net = random_network();
        let mut cache = RefreshCache::default();
        for fen in [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",
            "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let mut acc = Accumulator::default();
            acc.refresh(&net, &pos);
            check_tree(&net, &pos, &acc, 3, &mut cache);
        }
        cache.clear();
        let pos = Position::from_fen("8/8/2k5/8/2K5/8/3Q4/8 w - - 0 1").unwrap();
        let mut acc = Accumulator::default(); acc.refresh(&net, &pos);
        check_tree(&net, &pos, &acc, 4, &mut cache);
    }
}
