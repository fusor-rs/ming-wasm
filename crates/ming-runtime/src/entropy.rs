//! A deterministic getrandom backend (`--cfg getrandom_backend="custom"`, `.cargo/config.toml`):
//! only the tokenizer's hash maps ask for randomness, and fixed bytes keep
//! `crypto.getRandomValues` out of the module's imports.

#![expect(
    unsafe_code,
    reason = "getrandom's custom backend is an unmangled unsafe extern function"
)]

/// # Safety
/// getrandom 0.3's contract: `dest` is valid for writes of `len` bytes.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    // SAFETY: getrandom passes a buffer valid for `len` byte writes (the contract above).
    let out = unsafe { std::slice::from_raw_parts_mut(dest, len) };
    let mut state: u64 = 0x4D49_4E47_5454_5321;
    for byte in out {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    Ok(())
}
