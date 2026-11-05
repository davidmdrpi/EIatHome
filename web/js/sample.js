// The Live Lab's random sky sample: one real slice, picked at random, and what the sky map's
// marker card says about it.

import { deriveSeed, makeRng } from './rng.js';
import { T, driftToHzPerS, familyById } from './signals.js';
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

// The order the sample's frames are shown in with Shuffle frames on: one permutation per seed and
// slice, the same in the preview and in the training worker's waterfall.
export const sampleOrder = (seed, id) => makeRng(deriveSeed(seed, `sample-shuffle-${id}`)).permutation(T);

// The two halves of the draw: signals (detections and windows on a target's signal) and the
// controls they're compared with.
export const SIGNAL_FAMILIES = ['voyager', 'stars'];
export const CONTROL_FAMILIES = ['blind-window', 'offsource'];

// A random slice, never the one already shown (currentId). Half the draws are signals (any slice
// of Voyager 1 or the SETI targets), half are controls, split evenly between blind windows and
// off-source scans, however many slices each family has: featureless windows don't crowd out the
// signals. Slices with a pointing come first, so the map can show where the sample came from.
// null if there's nothing else to pick.
export function pickSample(slices, currentId = null, random = Math.random) {
  const others = slices.filter((s) => familyById[s.family]?.real && s.id !== currentId);
  const pool = others.some(located) ? others.filter(located) : others;
  if (!pool.length) return null;
  const pick = (list) => list[Math.min(list.length - 1, Math.floor(random() * list.length))];
  const signals = pool.filter((s) => !CONTROL_FAMILIES.includes(s.family));
  const controls = CONTROL_FAMILIES.map((f) => pool.filter((s) => s.family === f)).filter((list) => list.length);
  if (!controls.length) return pick(signals);
  if (!signals.length || random() >= 0.5) return pick(pick(controls));
  return pick(signals);
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
    // A window cut at a named frequency (blc1's, say) says so; otherwise the family's kind.
    window: source.window_type ?? family.window,
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

// The sample's name in captions: its target, centre frequency and telescope, as
// "Tau Ceti · 3285.618 MHz · Parkes".
export function sampleTitle(d) {
  const name = d.target ?? d.sourceName ?? familyById[d.family]?.label ?? d.id;
  return [name, d.freqMhz === null ? null : `${d.freqMhz.toFixed(3)} MHz`, d.telescope].filter(Boolean).join(' · ');
}

// What Section 1's sample card says under the title: drift, peak SNR, how the window was cut,
// and the slice's id.
export function sampleDetail(d) {
  return [
    d.driftHzPerS === null ? 'no drift measured' : `drift ${signed(d.driftHzPerS, 3)} Hz/s`,
    d.snr === null ? null : `peak SNR ${d.snr.toFixed(1)}`,
    d.window,
    d.id,
  ].filter(Boolean).join(' · ');
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
