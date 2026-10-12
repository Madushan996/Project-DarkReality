//! Generates slider attack tables (fancy magic bitboards) and square-relation
//! tables at build time, so the engine has no runtime initialisation step.

use std::{env, fmt::Write as _, fs, path::PathBuf};

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn on_board(rank: i32, file: i32) -> bool {
    (0..8).contains(&rank) && (0..8).contains(&file)
}

/// Attacks of a slider on `sq` given blockers `occ` (slow reference version).
fn slide(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    let mut attacks = 0;
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while on_board(r, f) {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occ & bit != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

/// Blocker squares that can influence the attack set (edges excluded).
fn relevant_mask(sq: usize, dirs: &[(i32, i32)]) -> u64 {
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    let mut mask = 0;
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while on_board(r + dr, f + df) {
            mask |= 1u64 << (r * 8 + f);
            r += dr;
            f += df;
        }
    }
    mask
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    table: Vec<u64>,
}

fn find_magic(sq: usize, dirs: &[(i32, i32)], rng: &mut Rng) -> Magic {
    let mask = relevant_mask(sq, dirs);
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits;

    let mut occs = Vec::with_capacity(size);
    let mut atts = Vec::with_capacity(size);
    let mut subset = 0u64;
    loop {
        occs.push(subset);
        atts.push(slide(sq, subset, dirs));
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }

    let mut table = vec![0u64; size];
    let mut epoch = vec![0u32; size];
    let mut attempt = 0u32;
    loop {
        let magic = rng.sparse();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        attempt += 1;
        let ok = occs.iter().zip(&atts).all(|(&occ, &att)| {
            let idx = (occ.wrapping_mul(magic) >> shift) as usize;
            if epoch[idx] != attempt {
                epoch[idx] = attempt;
                table[idx] = att;
                true
            } else {
                table[idx] == att
            }
        });
        if ok {
            return Magic { mask, magic, shift, table };
        }
    }
}

fn emit_magics(out: &mut String, name: &str, magics: &[Magic], offset: &mut usize) {
    writeln!(out, "pub static {name}: [MagicEntry; 64] = [").unwrap();
    for m in magics {
        writeln!(
            out,
            "    MagicEntry {{ mask: {:#x}, magic: {:#x}, shift: {}, offset: {} }},",
            m.mask, m.magic, m.shift, offset
        )
        .unwrap();
        *offset += m.table.len();
    }
    writeln!(out, "];").unwrap();
}

fn emit_square_table(out: &mut String, name: &str, f: impl Fn(usize, usize) -> u64) {
    writeln!(out, "pub static {name}: [[u64; 64]; 64] = [").unwrap();
    for a in 0..64 {
        out.push('[');
        for b in 0..64 {
            write!(out, "{:#x},", f(a, b)).unwrap();
        }
        out.push_str("],\n");
    }
    writeln!(out, "];").unwrap();
}

/// Embeds the network named by `DR_NET` (default `nets/current.nnue`). Without
/// one the engine builds with the material-only Gen0 evaluation.
fn embed_network() -> String {
    println!("cargo::rustc-check-cfg=cfg(dr_net)");
    println!("cargo:rerun-if-env-changed=DR_NET");
    let nets_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..").join("nets");
    let path = match env::var("DR_NET") {
        Ok(p) => PathBuf::from(p),
        Err(_) => nets_dir.join("current.nnue"),
    };
    // Watch the directory: a missing file would force a rebuild every time.
    println!("cargo:rerun-if-changed={}", if path.exists() { path.display() } else { nets_dir.display() });
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // The hidden-layer width is a compile-time constant taken from the
    // network header (bytes 12..16), so every loop has a fixed trip count.
    let mut hidden = 128u32;
    let mut buckets = 1u32;
    let mut king_buckets = 1u32;
    let mut version = 1u32;
    // Shown in the UCI `id name`; DR_NET_NAME overrides the file name.
    println!("cargo:rerun-if-env-changed=DR_NET_NAME");
    let mut net_name = "gen0".to_string();
    if path.exists() {
        let bytes = fs::read(&path).expect("read network");
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        // v1: inputs hidden qa qb scale; v2: inputs hidden buckets qa qb scale.
        match bytes.get(..8) {
            Some(b"DRNNUE01") => hidden = word(12),
            Some(b"DRNNUE02") => (hidden, buckets, version) = (word(12), word(16), 2),
            // v3: inputs hidden buckets king_buckets qa qb scale.
            Some(b"DRNNUE03") => (hidden, buckets, king_buckets, version) = (word(12), word(16), word(20), 3),
            _ => panic!("{} is not a DarkReality network", path.display()),
        }
        fs::write(out_dir.join("net.nnue"), bytes).expect("copy network");
        println!("cargo:rustc-cfg=dr_net");
        net_name = env::var("DR_NET_NAME")
            .unwrap_or_else(|_| path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
    }
    println!("cargo:rustc-env=DR_NET_NAME={net_name}");
    let arch = format!(
        "pub const HIDDEN: usize = {hidden};\npub const OUTPUT_BUCKETS: usize = {buckets};\npub const KING_BUCKETS: usize = {king_buckets};\npub const NET_VERSION: u32 = {version};\n"
    );
    fs::write(out_dir.join("net_arch.rs"), arch).unwrap();
    net_name
}

/// Finds a GNU resource compiler: DR_WINDRES, then PATH, then MSYS2.
fn find_windres() -> Option<PathBuf> {
    println!("cargo:rerun-if-env-changed=DR_WINDRES");
    if let Ok(p) = env::var("DR_WINDRES") {
        return Some(PathBuf::from(p));
    }
    let candidates = env::var_os("PATH")
        .map(|p| env::split_paths(&p).map(|d| d.join("windres.exe")).collect::<Vec<_>>())
        .unwrap_or_default();
    candidates
        .into_iter()
        .chain(["C:/msys64/ucrt64/bin/windres.exe", "C:/msys64/mingw64/bin/windres.exe"].map(PathBuf::from))
        .find(|p| p.exists())
}

/// Windows builds: embed the DarkReality icon and version information.
/// Skipped with a warning when no resource compiler is installed.
fn embed_windows_resources(net: &str) {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let icon = manifest.join("..").join("assets").join("darkreality.ico");
    println!("cargo:rerun-if-changed={}", icon.display());
    let Some(windres) = find_windres() else {
        println!("cargo:warning=windres not found; building without the icon (set DR_WINDRES)");
        return;
    };

    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let mut parts: Vec<&str> = version.split('.').collect();
    parts.resize(4, "0");
    let numeric = parts.join(",");
    let icon_path = icon.canonicalize().unwrap_or(icon).display().to_string().replace('\\', "/");
    let icon_path = icon_path.trim_start_matches("//?/");
    let rc = format!(
        r#"1 ICON "{icon_path}"
1 VERSIONINFO
FILEVERSION {numeric}
PRODUCTVERSION {numeric}
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "Madushan Dissanayake"
      VALUE "LegalCopyright", "Copyright (C) 2026 Madushan Dissanayake"
      VALUE "FileDescription", "DarkReality UCI chess engine"
      VALUE "FileVersion", "{version}"
      VALUE "InternalName", "darkreality"
      VALUE "OriginalFilename", "darkreality.exe"
      VALUE "ProductName", "DarkReality"
      VALUE "ProductVersion", "{version} {net}"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rc_path = out_dir.join("darkreality.rc");
    let obj = out_dir.join("darkreality-res.o");
    fs::write(&rc_path, rc).unwrap();

    // windres needs its own bin directory on PATH for its DLLs and cpp.
    let mut path = windres.parent().map(|p| p.as_os_str().to_owned()).unwrap_or_default();
    if let Some(old) = env::var_os("PATH") {
        path.push(";");
        path.push(old);
    }
    let status = std::process::Command::new(&windres)
        .args(["-O", "coff", "-i"])
        .arg(&rc_path)
        .arg("-o")
        .arg(&obj)
        .env("PATH", path)
        .status();
    match status {
        Ok(s) if s.success() => println!("cargo:rustc-link-arg-bins={}", obj.display()),
        _ => println!("cargo:warning=windres failed; building without the icon"),
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let net = embed_network();
    embed_windows_resources(&net);

    let mut rng = Rng(0x0DA2_C0DE_5EED_1234);
    let rooks: Vec<Magic> = (0..64).map(|sq| find_magic(sq, &ROOK_DIRS, &mut rng)).collect();
    let bishops: Vec<Magic> = (0..64).map(|sq| find_magic(sq, &BISHOP_DIRS, &mut rng)).collect();

    let mut out = String::new();
    let mut offset = 0;
    emit_magics(&mut out, "ROOK_MAGICS", &rooks, &mut offset);
    emit_magics(&mut out, "BISHOP_MAGICS", &bishops, &mut offset);

    writeln!(out, "pub static SLIDER_TABLE: [u64; {offset}] = [").unwrap();
    for (i, entry) in rooks.iter().chain(&bishops).flat_map(|m| &m.table).enumerate() {
        write!(out, "{entry:#x},").unwrap();
        if i % 16 == 15 {
            out.push('\n');
        }
    }
    writeln!(out, "];").unwrap();

    // Squares strictly between a and b when they share a line, else empty.
    emit_square_table(&mut out, "BETWEEN", |a, b| {
        for dirs in [&ROOK_DIRS, &BISHOP_DIRS] {
            if a != b && slide(a, 0, dirs) & (1 << b) != 0 {
                return slide(a, 1 << b, dirs) & slide(b, 1 << a, dirs);
            }
        }
        0
    });
    // The full edge-to-edge line through a and b when aligned, else empty.
    emit_square_table(&mut out, "LINE", |a, b| {
        for dirs in [&ROOK_DIRS, &BISHOP_DIRS] {
            if a != b && slide(a, 0, dirs) & (1 << b) != 0 {
                return (slide(a, 0, dirs) & slide(b, 0, dirs)) | (1 << a) | (1 << b);
            }
        }
        0
    });

    let path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("tables.rs");
    fs::write(path, out).unwrap();
}
