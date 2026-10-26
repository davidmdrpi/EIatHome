// Real-data "micro-slices": small windows of bandpass-flattened spectrogram (noise-sigma
// units) cut out of Breakthrough Listen files by tools/extract_slices.py. A manifest
// (index.json) describes each little-endian float32 .bin file.

import { T } from './signals.js';

export const MANIFEST_URL = 'data/slices/index.json';

function parseSlice(entry, buffer) {
  const [rows, width] = entry.shape;
  if (entry.dtype !== 'float32-le') throw new Error(`${entry.id}: unsupported dtype ${entry.dtype}`);
  if (rows !== T) throw new Error(`${entry.id}: needs ${T} time steps, has ${rows}`);
  if (buffer.byteLength !== rows * width * 4) throw new Error(`${entry.id}: file size doesn't match its shape`);
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
    driftChannelsPerStep: source.drift_channels_per_step,
    foffHz: source.foff_hz,
    tsamp: source.tsamp_s,
    sha256: entry.sha256 ?? null,
    source,
  };
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
      const name = entry.file.split('/').pop();
      const file = files.find((f) => f.name === name);
      if (!file) throw new Error(`Missing ${name}`);
      return parseSlice(entry, await file.arrayBuffer());
    }),
  );
}

// What workers need (typed arrays are copied by postMessage).
export const forWorkers = (slices) =>
  slices.map(({ id, group, width, data, driftChannelsPerStep, foffHz, tsamp }) => ({ id, group, width, data, driftChannelsPerStep, foffHz, tsamp }));
