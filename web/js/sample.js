// The Live Lab's random sky sample: one real slice, picked at random, and what the sky map's
// marker card says about it.

import { driftToHzPerS, familyById } from './signals.js';
import { mjdDate } from './splits.js';

// Whether a slice records where the telescope pointed (manifest version 2).
export const located = (slice) => Number.isFinite(slice?.source?.sky?.ra_hours) && Number.isFinite(slice?.source?.sky?.dec_deg);

// The sky map target a slice belongs to: one per family, pointing and date (see pointings() in
// main.js). null for a slice without a pointing.
export function pointingId(slice) {
  if (!located(slice)) return null;
  const sky = slice.source.sky;
  return `${slice.family}@${sky.ra_hours.toFixed(4)},${sky.dec_deg.toFixed(3)}@${String(sky.epoch ?? '').slice(0, 10)}`;
}

// A random slice, never the one already shown (currentId). Slices with a pointing come first,
// so the map can show where the sample came from. null if there's nothing else to pick.
export function pickSample(slices, currentId = null, random = Math.random) {
  const others = slices.filter((s) => familyById[s.family]?.real && s.id !== currentId);
  const pool = others.some(located) ? others.filter(located) : others;
  if (!pool.length) return null;
  return pool[Math.min(pool.length - 1, Math.floor(random() * pool.length))];
}

const signed = (v, digits) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`;

// What the marker card shows: the source, telescope, scan start, centre frequency, drift and
// SNR, and how the window was cut (on-target hit, off-source or random blind window).
export function describeSample(slice) {
  const source = slice.source ?? {};
  const sky = source.sky ?? {};
  const family = familyById[slice.family];
  const mjd = Number.isFinite(slice.startMjd) ? slice.startMjd : Number(source.tstart_mjd);
  const freq = Number(source.centre_freq_mhz);
  const recorded = Number(source.drift_hz_per_s);
  const fromChannels = driftToHzPerS(Number(source.drift_channels_per_step), Number(source.foff_hz), Number(source.tsamp_s));
  const drift = Number.isFinite(recorded) ? recorded : fromChannels;
  return {
    id: slice.id,
    family: slice.family,
    window: family.window,
    sourceName: source.source_name || null,
    target: sky.target ?? null,
    telescope: sky.telescope ?? source.archive?.telescope ?? null,
    mjd: Number.isFinite(mjd) ? mjd : null,
    date: Number.isFinite(mjd) ? mjdDate(mjd) : null,
    freqMhz: Number.isFinite(freq) ? freq : null,
    driftHzPerS: Number.isFinite(drift) ? drift : null,
    snr: Number.isFinite(Number(source.snr)) ? Number(source.snr) : null,
    ra_hours: located(slice) ? sky.ra_hours : null,
    dec_deg: located(slice) ? sky.dec_deg : null,
  };
}

// The card's two lines: what was observed, then the measurements.
export function sampleLines(d) {
  const window = d.window ? d.window[0].toUpperCase() + d.window.slice(1) : 'Slice';
  const name = d.target && d.sourceName && d.target.toLowerCase() !== d.sourceName.toLowerCase() ? `${d.target} (${d.sourceName})` : d.target ?? d.sourceName;
  const what = [window, name, d.telescope].filter(Boolean).join(' · ');
  const snr = d.snr === null ? null : d.driftHzPerS === null ? `peak SNR ${d.snr.toFixed(1)} in the window` : `SNR ${d.snr.toFixed(1)}`;
  const measured = [
    d.mjd === null ? null : `MJD ${d.mjd.toFixed(4)}`,
    d.freqMhz === null ? null : `${d.freqMhz.toFixed(6)} MHz`,
    d.driftHzPerS === null ? 'no drift measured' : `drift ${signed(d.driftHzPerS, 3)} Hz/s`,
    snr,
  ].filter(Boolean).join(' · ');
  return [what, measured];
}
