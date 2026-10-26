// Real-data "micro-slices": small windows of bandpass-flattened spectrogram (noise-sigma
// units) cut out of Breakthrough Listen files by tools/extract_slices.py. A manifest
// (index.json) describes each little-endian float32 .bin file.

import { REAL_FAMILIES, T } from './signals.js';

export const MANIFEST_URL = 'data/slices/index.json';

export async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error('Checking file hashes needs a secure page: open the Live Lab over https or from localhost.');
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// Validates one manifest entry against its bytes. The SHA-256 is always computed from the
// bytes; a digest in the manifest must match it.
export async function parseSlice(entry, buffer) {
  for (const key of ['id', 'group', 'file']) {
    if (typeof entry[key] !== 'string' || !entry[key]) throw new Error(`A slice is missing its "${key}".`);
  }
  if (!REAL_FAMILIES.includes(entry.family)) throw new Error(`${entry.id}: family must be one of ${REAL_FAMILIES.join(', ')}`);
  if (!Array.isArray(entry.shape) || entry.shape.length !== 2) throw new Error(`${entry.id}: shape must be [time steps, channels]`);
  const [rows, width] = entry.shape;
  if (entry.dtype !== 'float32-le') throw new Error(`${entry.id}: unsupported dtype ${entry.dtype}`);
  if (rows !== T) throw new Error(`${entry.id}: needs ${T} time steps, has ${rows}`);
  if (buffer.byteLength !== rows * width * 4) throw new Error(`${entry.id}: file size doesn't match its shape`);
  const sha256 = await sha256Hex(buffer);
  if (entry.sha256 != null && String(entry.sha256).toLowerCase() !== sha256) {
    throw new Error(`${entry.id}: the file's contents don't match the SHA-256 in the manifest`);
  }
  const view = new DataView(buffer);
  const data = new Float32Array(rows * width);
  for (let i = 0; i < data.length; i++) data[i] = view.getFloat32(i * 4, true);
  const source = entry.source ?? {};
  return {
    id: entry.id,
    family: entry.family,
    group: entry.group,
    width,
    data,
    driftChannelsPerStep: Number(source.drift_channels_per_step),
    foffHz: Number(source.foff_hz),
    tsamp: Number(source.tsamp_s),
    sha256,
    // When the scan started. Unlike a file name, this survives renaming and re-exporting,
    // so it's what the train/evaluation split uses (see splits.js). NaN if unknown.
    startMjd: source.tstart_mjd == null ? NaN : Number(source.tstart_mjd),
    source,
  };
}

const compare = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

// The one order every run uses, whatever order the manifest listed: by content, then id,
// then signal. Sampling indexes into this order, so reordering a manifest can't change results.
export function canonicalOrder(slices) {
  return [...slices].sort((a, b) => compare(a.sha256, b.sha256) || compare(a.id, b.id) || compare(a.group, b.group));
}

// One identity for a set of slices: their computed hashes plus the metadata that changes
// results (family, start time, resolution, drift labels), in canonical order.
export async function datasetId(slices) {
  const canonical = canonicalOrder(slices).map((s) => ({
    id: s.id,
    family: s.family,
    group: s.group,
    start_mjd: Number.isFinite(s.startMjd) ? s.startMjd : null,
    shape: [T, s.width],
    sha256: s.sha256,
    drift_channels_per_step: s.driftChannelsPerStep,
    foff_hz: s.foffHz,
    tsamp_s: s.tsamp,
  }));
  return sha256Hex(new TextEncoder().encode(JSON.stringify(canonical)));
}

// What a result card's slices are checked against: the dataset's identity and its families.
export async function describeDataset(slices) {
  if (!slices.length) return null;
  return { datasetId: await datasetId(slices), families: REAL_FAMILIES.filter((f) => slices.some((s) => s.family === f)) };
}

// Fetch the slices published with the site. Returns [] if there are none yet.
export async function loadPublishedSlices() {
  const response = await fetch(MANIFEST_URL, { cache: 'no-cache' });
  if (!response.ok) return [];
  const manifest = await response.json();
  const base = new URL(MANIFEST_URL, document.baseURI);
  const slices = await Promise.all(
    manifest.slices.map(async (entry) => {
      const bin = await fetch(new URL(entry.file, base));
      if (!bin.ok) throw new Error(`Couldn't fetch ${entry.file}`);
      return parseSlice(entry, await bin.arrayBuffer());
    }),
  );
  return canonicalOrder(slices);
}

// Load slices from files the user picked or dropped: one index.json plus its .bin files.
export async function loadSliceFiles(fileList) {
  const files = [...fileList];
  const manifestFile = files.find((f) => f.name.endsWith('.json'));
  if (!manifestFile) throw new Error('Include the index.json manifest along with the .bin files.');
  const manifest = JSON.parse(await manifestFile.text());
  if (manifest.format !== 'eiathome-slices') throw new Error(`${manifestFile.name} isn't an EI@Home slice manifest.`);
  const slices = await Promise.all(
    manifest.slices.map(async (entry) => {
      const name = String(entry.file).split('/').pop();
      const file = files.find((f) => f.name === name);
      if (!file) throw new Error(`Missing ${name}`);
      return parseSlice(entry, await file.arrayBuffer());
    }),
  );
  return canonicalOrder(slices);
}

// What workers need (typed arrays are copied by postMessage).
export const forWorkers = (slices) =>
  canonicalOrder(slices).map(({ id, family, group, width, data, driftChannelsPerStep, foffHz, tsamp, sha256, startMjd }) => ({
    id,
    family,
    group,
    width,
    data,
    driftChannelsPerStep,
    foffHz,
    tsamp,
    sha256,
    startMjd,
  }));
