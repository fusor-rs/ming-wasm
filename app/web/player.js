// Collects the engine's PCM, plays it through Web Audio while it is generated, and writes the
// whole utterance as a WAV.

const SAMPLE_RATE = 44100;
// Start streaming once this much audio is buffered, if generation runs faster than real time.
const STREAM_START_SECONDS = 1.0;
const STREAM_MIN_SPEED = 1.1;

/** Collects the PCM and plays it through Web Audio, chunk after chunk. */
export class Player {
  #context = new AudioContext({ sampleRate: SAMPLE_RATE });
  #chunks = [];
  #scheduled = 0;
  #next = 0;
  #started = false;
  #length = 0;

  push(samples) {
    this.#chunks.push(samples);
    this.#length += samples.length;
    if (this.#started) this.#schedule();
  }

  seconds() {
    return this.#length / SAMPLE_RATE;
  }

  /** Starts playback when it can no longer run dry (`speed`: audio seconds per wall second). */
  maybeStart(speed) {
    if (this.#started || !this.#length) return;
    if (speed === Infinity || (speed >= STREAM_MIN_SPEED && this.seconds() >= STREAM_START_SECONDS)) {
      this.#started = true;
      this.#next = this.#context.currentTime + 0.05;
      this.#schedule();
    }
  }

  #schedule() {
    for (; this.#scheduled < this.#chunks.length; this.#scheduled++) {
      const samples = this.#chunks[this.#scheduled];
      const buffer = this.#context.createBuffer(1, samples.length, SAMPLE_RATE);
      buffer.copyToChannel(samples, 0);
      const source = this.#context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.#context.destination);
      this.#next = Math.max(this.#next, this.#context.currentTime + 0.02);
      source.start(this.#next);
      this.#next += buffer.duration;
    }
  }

  close() {
    this.#context.close();
  }

  /** 16-bit PCM WAV of everything pushed. */
  wav() {
    const out = new DataView(new ArrayBuffer(44 + this.#length * 2));
    const text = (offset, s) => [...s].forEach((c, i) => out.setUint8(offset + i, c.charCodeAt(0)));
    text(0, 'RIFF');
    out.setUint32(4, 36 + this.#length * 2, true);
    text(8, 'WAVEfmt ');
    out.setUint32(16, 16, true);
    out.setUint16(20, 1, true);
    out.setUint16(22, 1, true);
    out.setUint32(24, SAMPLE_RATE, true);
    out.setUint32(28, SAMPLE_RATE * 2, true);
    out.setUint16(32, 2, true);
    out.setUint16(34, 16, true);
    text(36, 'data');
    out.setUint32(40, this.#length * 2, true);
    let offset = 44;
    for (const chunk of this.#chunks) {
      for (const s of chunk) {
        out.setInt16(offset, Math.round(Math.max(-1, Math.min(1, s)) * 32767), true);
        offset += 2;
      }
    }
    return out.buffer;
  }
}
