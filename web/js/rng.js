// Seeded random numbers, so every snippet, split and work unit is reproducible from its seed.

export function makeRng(seed) {
  let state = seed >>> 0;
  let spare = null;

  // mulberry32
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    int: (lo, hi) => lo + Math.floor(next() * (hi - lo)), // [lo, hi)
    logUniform: (lo, hi) => Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * next()),
    gauss() {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      let u = 0;
      while (u === 0) u = next();
      const r = Math.sqrt(-2 * Math.log(u));
      const angle = 2 * Math.PI * next();
      spare = r * Math.sin(angle);
      return r * Math.cos(angle);
    },
    permutation(n) {
      const p = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        [p[i], p[j]] = [p[j], p[i]];
      }
      return p;
    },
  };
}

// Derive an independent seed for a named purpose ("train", "eval", ...) from a base seed.
export function deriveSeed(seed, label) {
  let h = 2166136261 ^ (seed >>> 0);
  for (const ch of label) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
