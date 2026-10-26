// Real-data "micro-slices": small windows of bandpass-flattened spectrogram (noise-sigma
// units) cut out of Breakthrough Listen files by tools/extract_slices.py. A manifest
// (index.json) describes each little-endian float32 .bin file.

import { T } from './signals.js';

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
    // Slices from the same file and start time come from the same observation.
    observation: `${source.file ?? 'unknown'}@${source.tstart_mjd ?? 'unknown'}`,
    source,
  };
}

// One identity for a set of slices: their computed hashes plus the metadata that changes
// results (resolution and drift labels), in a canonical order.
export async function datasetId(slices) {
  const canonical = slices
    .map((s) => ({
      id: s.id,
      group: s.group,
      observation: s.observation,
      shape: [T, s.width],
      sha256: s.sha256,
      drift_channels_per_step: s.driftChannelsPerStep,
      foff_hz: s.foffHz,
      tsamp_s: s.tsamp,
    }))
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return sha256Hex(new TextEncoder().encode(JSON.stringify(canonical)));
}

// Fetch the slices published with the site. Returns [] if there are none yet.
export async function loadPublishedSlices() {
  const response = await fetch(MANIFEST_URL, { cache: 'no-cache' });
  if (!response.ok) return [];
  const manifest = await response.json();
  const base = new URL(MANIFEST_URL, document.baseURI);
  return Promise.all(
    manifest.slices.map(async (entry) => {
      const bin = await fetch(new URL(entry.file, base));
      if (!bin.ok) throw new Error(`Couldn't fetch ${entry.file}`);
      return parseSlice(entry, await bin.arrayBuffer());
    }),
  );
}

// Load slices from files the user picked or dropped: one index.json plus its .bin files.
export async function loadSliceFiles(fileList) {
  const files = [...fileList];
  const manifestFile = files.find((f) => f.name.endsWith('.json'));
  if (!manifestFile) throw new Error('Include the index.json manifest along with the .bin files.');
  const manifest = JSON.parse(await manifestFile.text());
  if (manifest.format !== 'eiathome-slices') throw new Error(`${manifestFile.name} isn't an EI@Home slice manifest.`);
  return Promise.all(
    manifest.slices.map(async (entry) => {
      const name = String(entry.file).split('/').pop();
      const file = files.find((f) => f.name === name);
      if (!file) throw new Error(`Missing ${name}`);
      return parseSlice(entry, await file.arrayBuffer());
    }),
  );
}

// What workers need (typed arrays are copied by postMessage).
export const forWorkers = (slices) =>
  slices.map(({ id, group, width, data, driftChannelsPerStep, foffHz, tsamp, sha256, observation }) => ({
    id,
    group,
    width,
    data,
    driftChannelsPerStep,
    foffHz,
    tsamp,
    sha256,
    observation,
  }));
