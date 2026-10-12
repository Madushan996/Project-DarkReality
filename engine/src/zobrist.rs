//! Zobrist hashing keys, generated at compile time from a fixed seed.

pub struct Keys {
    pub pieces: [[u64; 64]; 12],
    /// Indexed by the 4-bit castling-rights mask; each entry is the XOR of
    /// the keys of its set bits, so rights can be swapped in one XOR pair.
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub side: u64,
}

pub static KEYS: Keys = generate();

const fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn generate() -> Keys {
    let mut state = 0x0DA2_C0DE_2026_1002;
    let mut keys = Keys { pieces: [[0; 64]; 12], castling: [0; 16], ep_file: [0; 8], side: 0 };

    let mut p = 0;
    while p < 12 {
        let mut sq = 0;
        while sq < 64 {
            keys.pieces[p][sq] = splitmix(&mut state);
            sq += 1;
        }
        p += 1;
    }

    let mut base = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        base[i] = splitmix(&mut state);
        i += 1;
    }
    let mut mask = 0;
    while mask < 16 {
        let mut bit = 0;
        while bit < 4 {
            if mask & (1 << bit) != 0 {
                keys.castling[mask] ^= base[bit];
            }
            bit += 1;
        }
        mask += 1;
    }

    let mut f = 0;
    while f < 8 {
        keys.ep_file[f] = splitmix(&mut state);
        f += 1;
    }
    keys.side = splitmix(&mut state);
    keys
}
