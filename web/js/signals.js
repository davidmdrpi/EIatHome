// Signal families for the Live Lab: one real (Voyager 1, from pre-extracted slices) and
// several synthetic controls with known answers. Every generator returns a raw snippet in
// noise-sigma units (T time steps x C channels, row-major), before normalization.

export const T = 16; // time steps per snippet: all that Breakthrough Listen's high-resolution product has
export const C = 64; // channels per snippet

// Used to express drift rates in Hz/s when no real slice metadata is loaded:
// Breakthrough Listen's high-spectral-resolution product at Green Bank.
export const DEFAULT_FOFF_HZ = -2.7939677238464355;
export const DEFAULT_TSAMP_S = 18.253611008;

export const FAMILIES = [
  { id: 'voyager', label: 'Voyager 1', about: 'Real Green Bank data: Voyager 1’s carrier and telemetry sidebands. Needs extracted slices (see the Data tab).' },
  { id: 'tone', label: 'Drifting tone', about: 'A narrow line drifting at a constant rate, like a transmitter on a moving planet.' },
  { id: 'pulsed', label: 'Pulsed', about: 'A drifting tone switched on and off with a fixed period.' },
  { id: 'sweep', label: 'RFI sweep', about: 'A fast sawtooth sweep that wraps around the window, like swept radar interference.' },
  { id: 'rosetta', label: 'Rosetta (text)', about: 'English text sent as 16-tone FSK: one 4-bit symbol per time step. The positive control for “language-like”.' },
  { id: 'randfsk', label: 'Random FSK', about: 'Exactly the same modulation as Rosetta, carrying random symbols instead of text.' },
  { id: 'noise', label: 'Noise', about: 'Gaussian noise only. Nothing to learn.' },
];

export const familyById = Object.fromEntries(FAMILIES.map((f) => [f.id, f]));

// Public-domain English text for the Rosetta control.
export const ROSETTA_TEXT = `Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth.

We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America.`;

// UTF-8 bytes -> 4-bit symbols (high nibble, then low nibble). No compression, no scrambling.
export function textToSymbols(text) {
  const bytes = new TextEncoder().encode(text);
  const symbols = new Uint8Array(bytes.length * 2);
  bytes.forEach((b, i) => {
    symbols[2 * i] = b >> 4;
    symbols[2 * i + 1] = b & 15;
  });
  return symbols;
}

const FSK_TONES = 16;
const fskChannel = (symbol) => 8 + 3 * symbol; // tones 3 channels apart, channels 8..53

function noise(rng) {
  const x = new Float32Array(T * C);
  for (let i = 0; i < x.length; i++) x[i] = rng.gauss();
  return x;
}

// Add a narrow line whose position at (fractional) time step u is position(u), smeared by its
// motion during each step. `flux` is its total power per step (0 = off).
function addLine(x, position, flux, width = 0.6, nSub = 16) {
  const norm = nSub * width * Math.sqrt(2 * Math.PI);
  for (let t = 0; t < T; t++) {
    const f = flux(t);
    if (!f) continue;
    for (let s = 0; s < nSub; s++) {
      const centre = position(t + s / nSub);
      const lo = Math.max(0, Math.floor(centre - 4 * width));
      const hi = Math.min(C - 1, Math.ceil(centre + 4 * width));
      for (let c = lo; c <= hi; c++) {
        x[t * C + c] += (f * Math.exp(-0.5 * ((c - centre) / width) ** 2)) / norm;
      }
    }
  }
}

// Picks a symbol window. Track A splits the text into a training region and an evaluation
// region so the model can't simply memorize the evaluation text.
function symbolWindow(symbols, rng, region) {
  const split = Math.floor(symbols.length * 0.7);
  const [lo, hi] = region === 'train' ? [0, split - T] : [split, symbols.length - T];
  const start = rng.int(lo, hi + 1);
  return { window: symbols.subarray(start, start + T), start };
}

function fskSnippet(rng, symbols) {
  const x = noise(rng);
  const flux = rng.logUniform(30, 300);
  addLine(x, (u) => fskChannel(symbols[Math.min(T - 1, Math.floor(u))]), () => flux, 0.6, 4);
  return x;
}

let cachedText = null;
let cachedSymbols = null;
export function rosettaSymbols(text = ROSETTA_TEXT) {
  if (text !== cachedText) {
    cachedText = text;
    cachedSymbols = textToSymbols(text);
  }
  return cachedSymbols;
}

const GENERATORS = {
  noise: (rng) => ({ raw: noise(rng), label: {} }),

  tone(rng) {
    const x = noise(rng);
    const drift = rng.uniform(-3.5, 3.5); // channels per time step
    const flux = rng.logUniform(5, 1000);
    const centre = C / 2 + rng.uniform(-4, 4);
    addLine(x, (u) => centre + drift * (u - T / 2), () => flux);
    return { raw: x, label: { drift, flux } };
  },

  pulsed(rng) {
    const x = noise(rng);
    const drift = rng.uniform(-1.5, 1.5);
    const flux = rng.logUniform(20, 1000);
    const period = rng.int(3, 8);
    const on = rng.int(1, period);
    const phase = rng.int(0, period);
    const centre = C / 2 + rng.uniform(-4, 4);
    addLine(x, (u) => centre + drift * (u - T / 2), (t) => ((t + phase) % period < on ? flux : 0));
    return { raw: x, label: { drift, flux, period } };
  },

  sweep(rng) {
    const x = noise(rng);
    const rate = rng.uniform(8, 20) * (rng.next() < 0.5 ? -1 : 1);
    const flux = rng.logUniform(20, 1000);
    const start = rng.uniform(0, C);
    addLine(x, (u) => (((start + rate * u) % C) + C) % C, () => flux);
    return { raw: x, label: { rate, flux } };
  },

  rosetta(rng, ctx) {
    const { window, start } = symbolWindow(rosettaSymbols(ctx.text), rng, ctx.region);
    return { raw: fskSnippet(rng, window), label: { start } };
  },

  randfsk(rng) {
    const symbols = new Uint8Array(T).map(() => rng.int(0, FSK_TONES));
    return { raw: fskSnippet(rng, symbols), label: {} };
  },

  voyager(rng, ctx) {
    const slices = ctx.slices ?? [];
    if (!slices.length) throw new Error('No Voyager slices loaded');
    const slice = slices[rng.int(0, slices.length)];
    const offset = rng.int(-6, 7);
    const first = Math.round(slice.width / 2 - C / 2) + offset;
    const x = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
      x.set(slice.data.subarray(t * slice.width + first, t * slice.width + first + C), t * C);
    }
    return { raw: x, label: { drift: slice.driftChannelsPerStep, group: slice.group, slice: slice.id } };
  },
};

export function makeSnippet(family, rng, ctx = {}) {
  return GENERATORS[family](rng, ctx);
}

// A contiguous FSK stream for Track B: consecutive snippets carry consecutive symbols.
export function rosettaStream(rng, nSnippets, text) {
  const symbols = rosettaSymbols(text);
  const snippets = [];
  for (let i = 0; i < nSnippets; i++) {
    const start = (i * T) % Math.max(T, symbols.length - T);
    snippets.push(fskSnippet(rng, symbols.subarray(start, start + T)));
  }
  return { snippets, wrapped: nSnippets * T > symbols.length };
}

// Per-snippet robust normalization (median, then 1.4826 x median absolute deviation), then
// asinh to compress the huge dynamic range of bright carriers.
export function normalise(raw) {
  const sorted = Float32Array.from(raw).sort();
  const median = sorted[sorted.length >> 1];
  const deviations = sorted.map((v) => Math.abs(v - median)).sort();
  const scale = 1.4826 * deviations[deviations.length >> 1] || 1;
  return raw.map((v) => Math.asinh((v - median) / scale));
}

export function shuffleFrames(x, order) {
  const out = new Float32Array(x.length);
  order.forEach((src, t) => out.set(x.subarray(src * C, (src + 1) * C), t * C));
  return out;
}

export function driftToHzPerS(channelsPerStep, foffHz = DEFAULT_FOFF_HZ, tsamp = DEFAULT_TSAMP_S) {
  return (channelsPerStep * foffHz) / tsamp;
}
