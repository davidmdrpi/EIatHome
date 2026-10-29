// The Observatory Viewport: a sky map on a canvas that slews to each signal's position on the sky
// and zooms in while a computation runs.
//
// The sky is drawn from web/data/sky/sky.json (tools/build_sky.py): the 1,500 brightest Hipparcos
// stars coloured by B-V, every Hipparcos star near the observed targets (so a zoomed view shows
// the real field), constellation lines and names, and the IAU boundaries. The Milky Way is a band
// along the galactic plane. The projection is stereographic, which keeps stars round at any zoom.
//
// The math at the top has no DOM dependencies, so tests run it in Node. SkyMap below draws.

const DEG = Math.PI / 180;

// ---------- vectors and coordinates ----------

export function vec(raDeg, decDeg) {
  const a = raDeg * DEG;
  const d = decDeg * DEG;
  return [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)];
}

export function radec(v) {
  const ra = Math.atan2(v[1], v[0]) / DEG;
  return [(ra + 360) % 360, Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG];
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// Angle between two unit vectors, in degrees.
export const angleBetween = (a, b) => Math.atan2(Math.hypot(...cross(a, b)), dot(a, b)) / DEG;

// The point a fraction t of the way along the great circle from a to b.
export function slerp(a, b, t) {
  const omega = angleBetween(a, b) * DEG;
  if (omega < 1e-9) return a.slice();
  const s = Math.sin(omega);
  const wa = Math.sin((1 - t) * omega) / s;
  const wb = Math.sin(t * omega) / s;
  return norm([wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2]]);
}

// J2000 equatorial to galactic (Hipparcos introduction, vol. 1): rows are the galactic axes.
const GALACTIC = [
  [-0.0548755604162154, -0.873437090234885, -0.4838350155487132],
  [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
  [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
];

export function galacticToEquatorial(lDeg, bDeg) {
  const g = vec(lDeg, bDeg);
  return radec([0, 1, 2].map((i) => GALACTIC[0][i] * g[0] + GALACTIC[1][i] * g[1] + GALACTIC[2][i] * g[2]));
}

export function equatorialToGalactic(raDeg, decDeg) {
  const v = vec(raDeg, decDeg);
  return radec(GALACTIC.map((row) => dot(row, v)));
}

// Local sidereal time (hours) at a longitude (degrees east) for a date.
export function siderealHours(date, lonDeg) {
  const days = date.getTime() / 86400000 + 2440587.5 - 2451545.0;
  return (((18.697374558 + 24.06570982441908 * days + lonDeg / 15) % 24) + 24) % 24;
}

// Altitude and azimuth (degrees, azimuth from north through east) to RA (degrees) and Dec.
export function horizonToEquatorial(altDeg, azDeg, latDeg, lstHours) {
  const [a, A, phi] = [altDeg * DEG, azDeg * DEG, latDeg * DEG];
  const sinDec = Math.sin(a) * Math.sin(phi) + Math.cos(a) * Math.cos(phi) * Math.cos(A);
  const hourAngle = Math.atan2(-Math.sin(A) * Math.cos(a), Math.sin(a) * Math.cos(phi) - Math.cos(a) * Math.sin(phi) * Math.cos(A));
  return [(((lstHours * 15 - hourAngle / DEG) % 360) + 360) % 360, Math.asin(sinDec) / DEG];
}

// Green Bank Observatory, and the primary beam (FWHM) of its 100 m telescope: about 12.6' / f(GHz).
export const GREEN_BANK = { lat: 38.4331, lon: -79.8398 };
export const beamArcmin = (ghz) => 12.6 / ghz;

// ---------- constellations (IAU boundaries; tools/skycoords.py does the same) ----------

function unwrap(ring) {
  const pts = [ring[0].slice()];
  for (let i = 1; i < ring.length; i++) {
    const [lon, lat] = ring[i];
    pts.push([lon + 360 * Math.round((pts[pts.length - 1][0] - lon) / 360), lat]);
  }
  const [first, last] = [pts[0], pts[pts.length - 1]];
  const gap = last[0] === first[0] && last[1] === first[1] ? 0 : last[0] - first[0];
  if (Math.abs(Math.abs(gap) - 360) < 1) {
    const pole = pts.reduce((s, p) => s + p[1], 0) > 0 ? 90 : -90;
    pts.push([last[0], pole], [first[0], pole]);
  }
  return pts;
}

function inside(ra, dec, ring) {
  const pts = unwrap(ring);
  for (const x of [ra - 360, ra, ra + 360]) {
    let hits = false;
    for (let i = 0; i < pts.length; i++) {
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[(i + 1) % pts.length];
      if (y1 > dec !== y2 > dec && x < x1 + ((dec - y1) * (x2 - x1)) / (y2 - y1)) hits = !hits;
    }
    if (hits) return true;
  }
  return false;
}

// The constellation containing a position, or null if the boundaries are ambiguous there.
export function constellationAt(raDeg, decDeg, boundaries) {
  const found = boundaries.filter((b) => b.rings.some((ring) => inside(raDeg, decDeg, ring)));
  return found.length === 1 ? found[0].name : null;
}

// ---------- formatting ----------

const pad = (n, w = 2) => String(n).padStart(w, '0');

export function formatRA(hours) {
  let s = Math.round((((hours % 24) + 24) % 24) * 36000) / 10; // tenths of a second
  const h = Math.floor(s / 3600);
  s -= h * 3600;
  const m = Math.floor(s / 60);
  s -= m * 60;
  return `${pad(h % 24)}h ${pad(m)}m ${s.toFixed(1).padStart(4, '0')}s`;
}

export function formatDec(deg) {
  const sign = deg < 0 ? '−' : '+';
  let s = Math.round(Math.abs(deg) * 3600);
  const d = Math.floor(s / 3600);
  s -= d * 3600;
  const m = Math.floor(s / 60);
  return `${sign}${pad(d)}° ${pad(m)}′ ${pad(s - m * 60)}″`;
}

export const formatFov = (fov) => (fov >= 10 ? `${fov.toFixed(0)}°` : fov >= 1 ? `${fov.toFixed(1)}°` : `${(fov * 60).toFixed(0)}′`);

// ---------- star colours ----------

// B-V colour index to an RGB triple: temperature by Ballesteros (2012), then a blackbody
// approximation to sRGB (after Tanner Helland), softened toward white so stars stay readable.
export function bvToRgb(bv) {
  const b = Math.max(-0.4, Math.min(2.0, Number.isFinite(bv) ? bv : 0.6));
  const t = 4600 * (1 / (0.92 * b + 1.7) + 1 / (0.92 * b + 0.62)) / 100;
  const clamp = (x) => Math.max(0, Math.min(255, x));
  const r = t <= 66 ? 255 : clamp(329.698727446 * (t - 60) ** -0.1332047592);
  const g = t <= 66 ? clamp(99.4708025861 * Math.log(t) - 161.1195681661) : clamp(288.1221695283 * (t - 60) ** -0.0755148492);
  const bl = t >= 66 ? 255 : t <= 19 ? 0 : clamp(138.5177312231 * Math.log(t - 10) - 305.0447927307);
  return [r, g, bl].map((c) => Math.round(c * 0.75 + 255 * 0.25));
}

// ---------- the camera ----------

// A stereographic view centred on a unit vector, north up and east to the left (as the sky looks
// from below), with a vertical field of view of fov degrees on a width x height canvas.
export function camera(center, fov, width, height) {
  const f = norm(center);
  let e = cross([0, 0, 1], f);
  if (Math.hypot(...e) < 1e-6) e = [0, 1, 0]; // at a celestial pole
  e = norm(e);
  const n = cross(f, e);
  const scale = height / 2 / (2 * Math.tan((fov * DEG) / 4));
  const cx = width / 2;
  const cy = height / 2;
  return {
    center: f,
    fov,
    scale,
    width,
    height,
    // Screen position of a unit vector, or null if it's too far behind the view to draw.
    project(v) {
      const c = dot(v, f);
      if (c < -0.6) return null;
      const k = 2 / (1 + c);
      return [cx - scale * k * dot(v, e), cy - scale * k * dot(v, n)];
    },
    unproject(x, y) {
      const X = (cx - x) / scale;
      const Y = (cy - y) / scale;
      const r2 = X * X + Y * Y;
      return norm([0, 1, 2].map((i) => ((4 - r2) * f[i] + 4 * X * e[i] + 4 * Y * n[i]) / (4 + r2)));
    },
    // Pixels per degree at the centre of the view.
    pxPerDeg: scale * DEG,
  };
}

const smooth = (t) => t * t * (3 - 2 * t);

// A slew from one view to another: along the great circle between the centres, zooming out on
// the way when the move is longer than the view, as a camera crew would. Returns
// { duration (s), at(t) -> { center, fov } } for t from 0 to 1.
export function slew(fromCenter, fromFov, toCenter, toFov) {
  const angle = angleBetween(fromCenter, toCenter);
  const peak = Math.min(100, Math.max(fromFov, toFov, angle * 1.4));
  const duration = Math.min(3.2, 0.7 + angle / 45 + Math.abs(Math.log(toFov / fromFov)) / 4);
  return {
    duration,
    at(t) {
      const s = smooth(Math.max(0, Math.min(1, t)));
      const base = Math.exp(Math.log(fromFov) * (1 - s) + Math.log(toFov) * s);
      return { center: slerp(fromCenter, toCenter, s), fov: base + (peak - base) * Math.sin(Math.PI * s) * (peak > base ? 1 : 0) };
    },
  };
}

// The field of view while a computation is a fraction p done: from wide to the telescope beam,
// evenly in log scale.
export const WIDE_FOV = 60;
export const BEAM_FOV = 2;
export const computeFov = (p) => WIDE_FOV * (BEAM_FOV / WIDE_FOV) ** Math.max(0, Math.min(1, p));

// Recovery grid outcomes, by window (see recovery.js windowOutcomes).
export const OUTCOMES = ['empty', 'found', 'missed', 'false alarm', 'decoy flagged', 'decoy passed'];

// ---------- the viewport ----------

const STYLE = {
  bg: '#02040a',
  ink: '#e8eef8',
  dim: 'rgba(200, 215, 240, 0.62)',
  grid: 'rgba(120, 150, 210, 0.16)',
  gridLabel: 'rgba(150, 175, 225, 0.55)',
  lines: 'rgba(120, 170, 255, 0.32)',
  conLabel: 'rgba(150, 180, 240, 0.42)',
  milky: [170, 190, 255],
  signal: '#5cc8ff',
  off: '#ffb347',
  reference: '#c9a7ff',
  local: '#7ee081',
  horizon: 'rgba(126, 224, 129, 0.35)',
  beam: '#ffffff',
  outcome: {
    found: '#3ddc97',
    missed: '#ffb347',
    'false alarm': '#ff5d73',
    'decoy flagged': '#c9a7ff',
    'decoy passed': 'rgba(201, 167, 255, 0.28)',
    empty: 'rgba(120, 150, 210, 0.22)',
    scanned: 'rgba(92, 200, 255, 0.45)',
  },
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function glowSprite(rgb) {
  const size = 64;
  const c = new OffscreenCanvas(size, size);
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  const [r, gr, b] = rgb;
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.18, `rgba(${r},${gr},${b},0.95)`);
  grad.addColorStop(0.42, `rgba(${r},${gr},${b},0.28)`);
  grad.addColorStop(1, `rgba(${r},${gr},${b},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}

// Sky data, loaded once and shared by every viewport.
let skyPromise = null;
export function loadSky(url = new URL('../data/sky/sky.json', import.meta.url)) {
  skyPromise ??= fetch(url).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  });
  return skyPromise;
}

function prepareSky(sky) {
  const stars = [...sky.stars, ...sky.fields.flatMap((f) => f.stars)];
  const vecs = new Float64Array(stars.length * 3);
  const mags = new Float32Array(stars.length);
  const bins = new Uint8Array(stars.length);
  stars.forEach(([ra, dec, mag, bv], i) => {
    vecs.set(vec(ra, dec), i * 3);
    mags[i] = mag;
    bins[i] = Math.round((Math.max(-0.4, Math.min(2, bv ?? 0.6)) + 0.4) * 5); // 13 colour bins
  });
  const sprites = Array.from({ length: 13 }, (_, i) => glowSprite(bvToRgb(i / 5 - 0.4)));
  const lines = sky.constellations.flatMap((c) => c.lines.map((line) => line.map(([ra, dec]) => vec(ra, dec))));
  const labels = sky.constellations.map((c) => ({ name: c.name, rank: c.rank, v: vec(...c.label) }));
  const names = sky.names.map(([ra, dec, mag, name]) => ({ name, mag, v: vec(ra, dec) }));
  const milky = [];
  for (let l = 0; l < 360; l += 3) {
    const lc = l > 180 ? l - 360 : l; // brighter toward the galactic centre
    milky.push({ v: vec(...galacticToEquatorial(l, 0)), i: 0.35 + 0.65 * Math.exp(-((lc / 55) ** 2)), w: 7 + 5 * Math.exp(-((lc / 40) ** 2)) });
  }
  return { raw: sky, count: stars.length, vecs, mags, bins, sprites, lines, labels, names, milky };
}

// options: { title, onSelect(target), onObservatory(on), onAutoRun(on), autoRunLabel, onReview() }
export class SkyMap {
  constructor(container, options = {}) {
    this.options = options;
    this.reduceMotion = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.center = vec(258.2, 12.4);
    this.fov = WIDE_FOV;
    this.targets = [];
    this.selected = null;
    this.motion = null; // a slew in progress
    this.fovGoal = null; // where the zoom is easing to
    this.job = null; // the computation being shown
    this.sky = null;
    this.visible = true;
    this.frame = null;
    this.last = 0;
    this.build(container);
    loadSky()
      .then((sky) => {
        this.sky = prepareSky(sky);
        this.status('');
        this.request();
      })
      .catch((err) => this.status(`Couldn't load the sky (${err.message}).`));
  }

  // ----- DOM -----

  build(container) {
    const root = el('div', 'viewport');
    const canvas = el('canvas', 'sky');
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', 'Sky map');
    const target = el('div', 'hud-target');
    this.hud = {
      title: el('b', 'hud-title', this.options.title ?? 'Observatory'),
      place: el('span', 'hud-line'),
      detail: el('span', 'hud-line dim'),
      view: el('div', 'hud-view dim'),
      status: el('div', 'hud hud-status dim', 'Loading the sky…'),
      telemetry: el('div', 'hud-telemetry'),
      source: el('div', 'dim'),
      line: el('span', 'hud-line'),
      bar: el('i'),
      legend: el('div', 'hud-legend'),
      stats: el('div', 'hud-stats dim'),
    };
    target.append(this.hud.title, this.hud.place, this.hud.detail);
    const progress = el('div', 'hud-progress');
    progress.append(this.hud.bar);
    this.hud.telemetry.setAttribute('role', 'status');
    this.hud.telemetry.setAttribute('aria-live', 'polite');
    this.hud.telemetry.append(this.hud.source, this.hud.line, progress);
    this.hud.telemetry.hidden = true;
    this.hud.legend.hidden = true;

    const controls = el('div', 'hud-controls');
    const wide = el('button', 'ghost', 'Wide view');
    wide.type = 'button';
    wide.onclick = () => this.goTo(this.selected ?? { ra_hours: radec(this.center)[0] / 15, dec_deg: radec(this.center)[1] }, WIDE_FOV);
    const observatory = el('button', 'ghost obs-toggle', 'Observatory mode');
    observatory.type = 'button';
    observatory.setAttribute('aria-pressed', 'false');
    observatory.onclick = () => this.setObservatory(!this.observatory);
    const auto = el('label', 'check auto-run');
    this.autoBox = el('input');
    this.autoBox.type = 'checkbox';
    this.autoBox.onchange = () => this.options.onAutoRun?.(this.autoBox.checked);
    auto.append(this.autoBox, document.createTextNode(` ${this.options.autoRunLabel ?? 'Auto-run next work unit'}`));
    auto.hidden = true;
    this.autoLabel = auto;
    this.obsButton = observatory;
    // Opens the page's review of the results computed in this browser (setReview labels it).
    const review = el('button', 'ghost review', '');
    review.type = 'button';
    review.hidden = true;
    review.onclick = () => this.options.onReview?.();
    this.reviewButton = review;
    controls.append(review, wide, observatory, auto);

    // The HUD's corners flow in two bars, so they wrap rather than overlap on narrow screens:
    // the target and the view along the top, the readout and the controls along the bottom.
    const top = el('div', 'hud hud-bar hud-top');
    const corner = el('div', 'hud-corner');
    corner.append(this.hud.view, this.hud.stats);
    top.append(target, corner);
    const bottom = el('div', 'hud hud-bar hud-bottom');
    const readout = el('div', 'hud-readout');
    readout.append(this.hud.legend, this.hud.telemetry);
    bottom.append(readout, controls);
    root.append(canvas, top, this.hud.status, bottom);
    container.append(root);
    this.root = root;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.bindInput();
    new ResizeObserver(() => this.resize()).observe(canvas);
    if (typeof IntersectionObserver === 'function') {
      new IntersectionObserver(([entry]) => {
        this.visible = entry.isIntersecting;
        if (this.visible) this.request();
      }).observe(canvas);
    }
    document.addEventListener('visibilitychange', () => !document.hidden && this.request());
    // Escape leaves observatory mode, unless it's closing a dialog opened over it.
    document.addEventListener('keydown', (e) => e.key === 'Escape' && this.observatory && !document.querySelector('dialog[open]') && this.setObservatory(false));
    this.resize();
  }

  status(text) {
    this.hud.status.textContent = text;
    this.hud.status.hidden = !text;
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.request();
  }

  bindInput() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      drag = { x: e.offsetX, y: e.offsetY, moved: 0, cam: this.cam(), center: this.center };
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag) return this.hover(e.offsetX, e.offsetY);
      drag.moved = Math.max(drag.moved, Math.hypot(e.offsetX - drag.x, e.offsetY - drag.y));
      if (drag.moved < 3) return;
      // Keep the point grabbed under the pointer: rotate the view by the grabbed point's move.
      const from = drag.cam.unproject(drag.x, drag.y);
      const to = drag.cam.unproject(e.offsetX, e.offsetY);
      const axis = norm(cross(to, from));
      const angle = angleBetween(to, from) * DEG;
      this.center = rotate(drag.center, axis, angle);
      this.motion = null;
      this.request();
    });
    const end = (e) => {
      if (drag && drag.moved < 3) this.click(e.offsetX, e.offsetY);
      drag = null;
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', () => (drag = null));
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.motion = null;
      this.fovGoal = null;
      this.fov = Math.max(0.5, Math.min(120, this.fov * Math.exp(e.deltaY * 0.0015)));
      this.request();
    }, { passive: false });
    c.addEventListener('keydown', (e) => {
      const step = this.fov / 10;
      const [ra, dec] = radec(this.center);
      const moves = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
      if (moves[e.key]) {
        const [dra, ddec] = moves[e.key];
        this.center = vec(ra + dra / Math.max(0.2, Math.cos(dec * DEG)), Math.max(-89.9, Math.min(89.9, dec + ddec)));
      } else if (e.key === '+' || e.key === '=') this.fov = Math.max(0.5, this.fov / 1.25);
      else if (e.key === '-') this.fov = Math.min(120, this.fov * 1.25);
      else return;
      e.preventDefault();
      this.motion = null;
      this.fovGoal = null;
      this.request();
    });
  }

  cam() {
    return camera(this.center, this.fov, this.canvas.clientWidth || 1, this.canvas.clientHeight || 1);
  }

  // The target nearest a screen position, within 16 pixels.
  pick(x, y) {
    const cam = this.cam();
    let best = null;
    let bestD = 16;
    for (const t of this.targets) {
      const p = cam.project(t.v);
      const d = p && Math.hypot(p[0] - x, p[1] - y);
      if (p && d < bestD) [best, bestD] = [t, d];
    }
    return best;
  }

  hover(x, y) {
    this.canvas.style.cursor = this.pick(x, y) ? 'pointer' : 'grab';
  }

  click(x, y) {
    const t = this.pick(x, y);
    if (!t) return;
    this.select(t.id);
    this.options.onSelect?.(t);
  }

  setObservatory(on) {
    this.observatory = on;
    this.root.classList.toggle('observatory', on);
    this.obsButton.setAttribute('aria-pressed', String(on));
    this.obsButton.textContent = on ? 'Exit observatory' : 'Observatory mode';
    this.autoLabel.hidden = !on;
    if (!on && this.autoBox.checked) {
      this.autoBox.checked = false;
      this.options.onAutoRun?.(false);
    }
    this.options.onObservatory?.(on);
    requestAnimationFrame(() => this.resize());
  }

  setStats(text) {
    this.hud.stats.textContent = text;
  }

  // The review button's label, or null to hide it; attention when results await submission.
  setReview(text, attention = false) {
    this.reviewButton.hidden = !text;
    this.reviewButton.textContent = text ?? '';
    this.reviewButton.classList.toggle('attention', attention);
  }

  // Ticks or unticks the auto-run box without calling onAutoRun (the page changed it).
  setAutoRun(on) {
    this.autoBox.checked = on;
  }

  // ----- targets and motion -----

  // targets: [{ id, label, kind: 'signal' | 'off' | 'reference' | 'local', ra_hours, dec_deg,
  //   constellation?, distance_label?, epoch?, telescope?, note?, family? }]
  setTargets(targets) {
    this.targets = targets.map((t) => ({ ...t, v: vec(t.ra_hours * 15, t.dec_deg) }));
    if (this.selected) this.selected = this.targets.find((t) => t.id === this.selected.id) ?? null;
    this.request();
  }

  target(id) {
    return this.targets.find((t) => t.id === id) ?? null;
  }

  // Selects a target, shows it in the HUD and slews to it.
  select(id, fov = Math.min(this.fov, 20)) {
    const t = this.target(id);
    if (!t) return;
    this.selected = t;
    this.describe(t);
    this.goTo(t, fov);
  }

  goTo(where, fov) {
    const to = where.v ?? vec(where.ra_hours * 15, where.dec_deg);
    this.fovGoal = null;
    if (this.reduceMotion) {
      [this.center, this.fov] = [to, fov];
      this.motion = null;
    } else {
      this.motion = { plan: slew(this.center, this.fov, to, fov), start: performance.now() };
    }
    this.request();
  }

  describe(t) {
    this.hud.title.textContent = t.label;
    const where = [t.constellation, `RA ${formatRA(t.ra_hours)}`, `Dec ${formatDec(t.dec_deg)}`].filter(Boolean).join(' · ');
    this.hud.place.textContent = where;
    const date = t.epoch && t.epoch !== 'J2000' ? t.epoch.slice(0, 10) : t.epoch;
    this.hud.detail.textContent = [t.distance_label, t.telescope, date, t.note].filter(Boolean).join(' · ');
    this.canvas.setAttribute('aria-label', `Sky map centred on ${t.label}: ${where}`);
  }

  // ----- computations -----

  // kind: 'train' | 'unit' | 'tokens' | 'analysis' | 'recovery'. The view slews to the target,
  // wide, then zooms in with progress. A job without progress ('analysis') keeps the framing.
  // The job stays on the target whose data it computes on: selecting another target while it
  // runs moves the view, not the job. source describes the data when the target's own label
  // doesn't say all of it (data from several sessions, say).
  startJob(kind, targetId, title, source) {
    const t = this.target(targetId) ?? this.selected;
    if (t && kind !== 'analysis') {
      this.selected = t;
      this.describe(t);
      this.goTo(t, WIDE_FOV);
    }
    this.job = { kind, title, target: t, progress: kind === 'analysis' ? null : 0, started: performance.now(), done: false, rings: [], grid: null };
    this.hud.source.textContent = source ?? (t ? `Data from ${t.label}` : '');
    if (kind === 'recovery') this.job.grid = { states: new Uint8Array(1024), scanned: 0, outcomes: null, reveal: 0, pings: [] };
    this.hud.telemetry.hidden = false;
    this.hud.legend.hidden = kind !== 'recovery';
    this.hud.legend.textContent = kind === 'recovery' ? 'Dealing 1,024 windows…' : '';
    this.telemetry(title, this.job.progress === null ? undefined : 0); // no zoom without progress
  }

  telemetry(text, progress) {
    if (!this.job) return;
    if (progress !== undefined && progress !== null) this.job.progress = Math.max(this.job.progress ?? 0, Math.min(1, progress));
    this.hud.line.textContent = text;
    this.hud.bar.style.width = `${Math.round((this.job.progress ?? 0) * 100)}%`;
    if (this.following() && !this.motion) this.fovGoal = computeFov(this.job.progress);
    this.request();
  }

  // Whether the view zooms with the job's progress: only while it's on the job's target.
  following() {
    return Boolean(this.job) && this.job.progress !== null && Boolean(this.selected) && this.selected.id === this.job.target?.id;
  }

  // Recovery: the background windows (cells, by position along the band) just scored. They come
  // in the shuffled order the dealer handed them out.
  scanned(cells) {
    const grid = this.job?.grid;
    if (!grid || !cells.length) return;
    for (const c of cells) grid.states[c] = 1;
    grid.pings.push({ cell: cells[cells.length - 1], at: performance.now() });
    grid.scanned += cells.length;
    this.request();
  }

  // Recovery: each window's outcome (indices into OUTCOMES), revealed in a sweep across the grid.
  outcomes(codes, legend, title = '') {
    const grid = this.job?.grid;
    if (!grid) return;
    grid.outcomes = Uint8Array.from(codes);
    grid.revealStart = performance.now();
    this.hud.legend.replaceChildren(
      ...(title ? [el('span', 'legend-title', title)] : []),
      ...legend.map(({ outcome, count }) => {
        const item = el('span', 'legend-item');
        const key = el('span', 'key');
        key.style.background = STYLE.outcome[outcome];
        item.append(key, document.createTextNode(`${outcome} ${count}`));
        return item;
      }),
    );
    this.request();
  }

  finishJob(text) {
    if (!this.job) return;
    this.job.done = true;
    this.job.finished = performance.now();
    if (this.job.progress !== null) this.job.progress = 1;
    this.telemetry(text, this.job.progress ?? undefined);
    if (this.job.progress === null) this.hud.bar.style.width = '100%';
  }

  // ----- drawing -----

  request() {
    if (this.frame === null) this.frame = requestAnimationFrame((t) => this.draw(t));
  }

  animating(now) {
    return Boolean(this.motion) || this.fovGoal !== null || (this.job && (!this.job.done || now - (this.job.finished ?? now) < 3000));
  }

  draw(now) {
    this.frame = null;
    const dt = Math.min(0.1, (now - (this.last || now)) / 1000);
    this.last = now;
    if (!this.visible || document.hidden || !this.canvas.clientWidth) return;

    if (this.motion) {
      const t = (now - this.motion.start) / 1000 / this.motion.plan.duration;
      const view = this.motion.plan.at(t);
      [this.center, this.fov] = [view.center, view.fov];
      if (t >= 1) {
        this.motion = null;
        if (this.following()) this.fovGoal = computeFov(this.job.progress);
      }
    } else if (this.fovGoal !== null) {
      const k = this.reduceMotion ? 1 : 1 - Math.exp(-dt * 2.5);
      this.fov = Math.exp(Math.log(this.fov) + (Math.log(this.fovGoal) - Math.log(this.fov)) * k);
      if (Math.abs(Math.log(this.fov / this.fovGoal)) < 0.002) [this.fov, this.fovGoal] = [this.fovGoal, null];
    }

    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    const cam = this.cam();
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = STYLE.bg;
    ctx.fillRect(0, 0, w, h);
    if (this.sky) {
      this.drawMilkyWay(ctx, cam);
      this.drawGrid(ctx, cam);
      this.drawConstellations(ctx, cam);
      this.drawStars(ctx, cam);
      this.drawHorizon(ctx, cam);
      this.drawJob(ctx, cam, now);
      this.drawTargets(ctx, cam, now);
    }
    const [ra, dec] = radec(this.center);
    const here = this.sky ? constellationAt(ra, dec, this.sky.raw.boundaries) : null;
    this.hud.view.textContent = `FOV ${formatFov(this.fov)} · ${formatRA(ra / 15)} ${formatDec(dec)}${here ? ` · ${here}` : ''}`;
    this.canvas.dataset.center = `${ra.toFixed(4)},${dec.toFixed(4)}`;
    this.canvas.dataset.fov = this.fov.toFixed(3);
    if (this.animating(now)) this.request();
  }

  // A polyline of unit vectors, broken where it leaves the drawable part of the sphere.
  path(ctx, cam, points) {
    let open = false;
    let prev = null;
    const limit = Math.hypot(cam.width, cam.height) * 2;
    for (const v of points) {
      const p = cam.project(v);
      if (!p || Math.abs(p[0] - cam.width / 2) > limit || Math.abs(p[1] - cam.height / 2) > limit || (prev && Math.hypot(p[0] - prev[0], p[1] - prev[1]) > limit / 2)) {
        open = false;
        prev = null;
        continue;
      }
      if (open) ctx.lineTo(p[0], p[1]);
      else ctx.moveTo(p[0], p[1]);
      open = true;
      prev = p;
    }
  }

  drawMilkyWay(ctx, cam) {
    const [r, g, b] = STYLE.milky;
    ctx.globalCompositeOperation = 'lighter';
    for (const m of this.sky.milky) {
      const p = cam.project(m.v);
      if (!p) continue;
      const radius = m.w * cam.pxPerDeg;
      if (radius > 6 * Math.max(cam.width, cam.height) || p[0] < -radius || p[0] > cam.width + radius || p[1] < -radius || p[1] > cam.height + radius) continue;
      const grad = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], radius);
      grad.addColorStop(0, `rgba(${r},${g},${b},${0.075 * m.i})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(p[0] - radius, p[1] - radius, radius * 2, radius * 2);
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  drawGrid(ctx, cam) {
    const fov = cam.fov;
    const decStep = [30, 15, 10, 5, 2, 1, 0.5, 0.25, 1 / 6].find((s) => fov / s >= 3) ?? 1 / 6;
    const raStep = [30, 15, 7.5, 3.75, 2.5, 1.25, 0.5, 0.25, 1 / 6].find((s) => fov / s >= 3) ?? 1 / 6;
    const [ra0, dec0] = radec(cam.center);
    const span = Math.min(180, fov * 1.6);
    const sample = Math.max(0.05, fov / 60);
    ctx.strokeStyle = STYLE.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let dec = Math.ceil((dec0 - span) / decStep) * decStep; dec <= dec0 + span; dec += decStep) {
      if (Math.abs(dec) >= 90) continue;
      const raSpan = fov > 60 ? 180 : Math.min(180, span / Math.max(0.05, Math.cos(dec * DEG)));
      const pts = [];
      for (let ra = ra0 - raSpan; ra <= ra0 + raSpan; ra += sample / Math.max(0.05, Math.cos(dec * DEG))) pts.push(vec(ra, dec));
      this.path(ctx, cam, pts);
    }
    const raSpan = fov > 60 ? 180 : Math.min(180, span / Math.max(0.05, Math.cos(dec0 * DEG)));
    for (let ra = Math.ceil((ra0 - raSpan) / raStep) * raStep; ra <= ra0 + raSpan; ra += raStep) {
      const pts = [];
      for (let dec = Math.max(-89, dec0 - span); dec <= Math.min(89, dec0 + span); dec += sample) pts.push(vec(ra, dec));
      this.path(ctx, cam, pts);
    }
    ctx.stroke();
  }

  drawConstellations(ctx, cam) {
    const alpha = Math.max(0, Math.min(1, (cam.fov - 3) / 12));
    if (!alpha) return;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = STYLE.lines;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const line of this.sky.lines) this.path(ctx, cam, line);
    ctx.stroke();
    if (cam.fov > 18) {
      ctx.fillStyle = STYLE.conLabel;
      ctx.font = '10px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      for (const c of this.sky.labels) {
        if (cam.fov < 40 && c.rank > 2) continue;
        const p = cam.project(c.v);
        if (p && p[0] > 0 && p[0] < cam.width && p[1] > 0 && p[1] < cam.height) ctx.fillText(c.name.toUpperCase(), p[0], p[1]);
      }
    }
    ctx.globalAlpha = 1;
  }

  drawStars(ctx, cam) {
    const { vecs, mags, bins, sprites, count } = this.sky;
    // Fainter stars appear as the view narrows, as through a telescope.
    const limit = Math.min(11.5, 4.9 + 3.2 * Math.log10(Math.max(1, WIDE_FOV / cam.fov)));
    const reach = Math.cos(Math.min(170, cam.fov * (1 + cam.width / cam.height)) * 0.6 * DEG);
    const f = cam.center;
    const v = [0, 0, 0];
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < count; i++) {
      const mag = mags[i];
      if (mag > limit) continue;
      v[0] = vecs[i * 3];
      v[1] = vecs[i * 3 + 1];
      v[2] = vecs[i * 3 + 2];
      if (v[0] * f[0] + v[1] * f[1] + v[2] * f[2] < reach) continue;
      const p = cam.project(v);
      if (!p || p[0] < -8 || p[1] < -8 || p[0] > cam.width + 8 || p[1] > cam.height + 8) continue;
      const size = Math.max(1.6, 2.4 * (limit + 1.2 - mag));
      ctx.globalAlpha = Math.min(1, 0.35 + 0.18 * (limit + 1 - mag));
      ctx.drawImage(sprites[bins[i]], p[0] - size, p[1] - size, size * 2, size * 2);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    if (cam.fov < 100) {
      ctx.fillStyle = STYLE.dim;
      ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      for (const s of this.sky.names) {
        const p = cam.project(s.v);
        if (p && p[0] > 0 && p[0] < cam.width - 40 && p[1] > 10 && p[1] < cam.height) ctx.fillText(s.name, p[0] + 7, p[1] - 5);
      }
    }
  }

  drawHorizon(ctx, cam) {
    const local = this.targets.find((t) => t.kind === 'local');
    if (!local || cam.fov < 20) return;
    const lst = siderealHours(new Date(), GREEN_BANK.lon);
    const pts = [];
    for (let az = 0; az <= 360; az += 3) pts.push(vec(...horizonToEquatorial(0, az, GREEN_BANK.lat, lst)));
    ctx.strokeStyle = STYLE.horizon;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    this.path(ctx, cam, pts);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawTargets(ctx, cam, now) {
    const pulse = this.reduceMotion ? 0.5 : (Math.sin(now / 420) + 1) / 2;
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    const shown = [];
    const placed = []; // label boxes drawn so far: a label that would overlap one is left out
    const rank = (t) => (t === this.selected ? 0 : { signal: 1, off: 2, local: 3 }[t.kind] ?? 4);
    for (const t of [...this.targets].sort((a, b) => rank(a) - rank(b))) {
      const p = cam.project(t.v);
      if (!p || p[0] < -40 || p[1] < -40 || p[0] > cam.width + 40 || p[1] > cam.height + 40) continue;
      if (p[0] >= 0 && p[1] >= 0 && p[0] <= cam.width && p[1] <= cam.height) shown.push(`${t.id} ${Math.round(p[0])} ${Math.round(p[1])}`);
      const color = STYLE[t.kind] ?? STYLE.signal;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (t.kind === 'reference') {
        ctx.moveTo(p[0], p[1] - 6);
        ctx.lineTo(p[0] + 6, p[1]);
        ctx.lineTo(p[0], p[1] + 6);
        ctx.lineTo(p[0] - 6, p[1]);
        ctx.closePath();
      } else if (t.kind === 'local') {
        ctx.moveTo(p[0], p[1] - 9);
        ctx.lineTo(p[0] - 6, p[1] + 6);
        ctx.lineTo(p[0] + 6, p[1] + 6);
        ctx.closePath();
        ctx.moveTo(p[0] - 9, p[1] + 9);
        ctx.lineTo(p[0] + 9, p[1] + 9);
      } else {
        if (t.kind === 'off') ctx.setLineDash([3, 3]);
        ctx.arc(p[0], p[1], 6 + (t === this.selected ? 0 : 2 * pulse), 0, 2 * Math.PI);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (t.kind !== 'reference' && t.kind !== 'local') {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 1.8, 0, 2 * Math.PI);
        ctx.fill();
      }
      const text = t.short ?? t.label;
      const box = [p[0] + 10, p[1] - 8, ctx.measureText(text).width, 14];
      if (placed.some((o) => box[0] < o[0] + o[2] && o[0] < box[0] + box[2] && box[1] < o[1] + o[3] && o[1] < box[1] + box[3])) continue;
      placed.push(box);
      ctx.fillText(text, p[0] + 10, p[1] + 4);
    }
    // Where each visible beacon is, in CSS pixels: for tests and assistive tooling.
    this.canvas.dataset.beacons = shown.join(';');
    if (this.selected) this.drawReticle(ctx, cam, this.selected);
  }

  // The Green Bank Telescope's primary beam at 8.4 GHz around the selected target, drawn at
  // least large enough to see.
  drawReticle(ctx, cam, t) {
    const p = cam.project(t.v);
    if (!p) return;
    const beamDeg = beamArcmin(8.42) / 60;
    const true_ = (beamDeg / 2) * cam.pxPerDeg;
    const r = Math.max(14, true_);
    ctx.strokeStyle = STYLE.beam;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(p[0] + dx * (r + 4), p[1] + dy * (r + 4));
      ctx.lineTo(p[0] + dx * (r + 14), p[1] + dy * (r + 14));
    }
    ctx.stroke();
    if (true_ >= 14) {
      ctx.fillStyle = STYLE.beam;
      ctx.textAlign = 'center';
      ctx.fillText(`GBT beam ${beamArcmin(8.42).toFixed(1)}′ at 8.4 GHz`, p[0], p[1] + r + 28);
    }
    ctx.globalAlpha = 1;
  }

  drawJob(ctx, cam, now) {
    const job = this.job;
    const t = job?.target;
    this.canvas.dataset.job = t?.id ?? ''; // where the job is drawn: for tests
    if (!job || !t) return;
    const p = cam.project(t.v);
    if (!p) return;
    const age = (now - job.started) / 1000;
    if (job.kind !== 'recovery' && !(job.done && now - job.finished > 2500)) {
      // Wavefronts from the source while it computes; they stop after it finishes.
      const maxR = Math.min(cam.width, cam.height) * 0.42;
      const period = 1.1;
      ctx.lineWidth = 1.5;
      for (let k = 0; k < 4; k++) {
        const phase = this.reduceMotion ? k / 4 : ((age / period + k / 4) % 1);
        if (job.done && phase > 0.95) continue;
        ctx.strokeStyle = `rgba(92, 200, 255, ${0.55 * (1 - phase)})`;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 10 + phase * maxR, 0, 2 * Math.PI);
        ctx.stroke();
      }
      // A drifting carrier: a short streak sweeping slowly around the beam.
      const a = this.reduceMotion ? 0 : age * 0.8;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.beginPath();
      ctx.arc(p[0], p[1], 22, a, a + 0.9);
      ctx.stroke();
    }
    if (job.grid) this.drawGridScan(ctx, cam, p, now);
  }

  // Recovery: the 1,024 frequency windows of the one off-source pointing, laid out as an
  // illustrative 32 x 32 grid around it (the cells aren't positions on the sky),
  // in order along the band. Cells light as their window is scored, then show its outcome.
  drawGridScan(ctx, cam, p, now) {
    const grid = this.job.grid;
    const side = Math.max(64, Math.min(Math.min(cam.width, cam.height) * 0.6, 1.6 * cam.pxPerDeg));
    const cell = side / 32;
    const x0 = p[0] - side / 2;
    const y0 = p[1] - side / 2;
    const revealed = grid.outcomes ? Math.min(1024, Math.floor(((now - grid.revealStart) / 1400) * 1024)) : 0;
    for (let i = 0; i < 1024; i++) {
      let color = null;
      if (i < revealed) color = STYLE.outcome[OUTCOMES[grid.outcomes[i]]];
      else if (grid.states[i]) color = STYLE.outcome.scanned;
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.fillRect(x0 + (i % 32) * cell + 0.5, y0 + Math.floor(i / 32) * cell + 0.5, Math.max(1, cell - 1), Math.max(1, cell - 1));
    }
    ctx.strokeStyle = 'rgba(92, 200, 255, 0.5)';
    ctx.strokeRect(x0, y0, side, side);
    // Pings: a ring where scoring has reached.
    grid.pings = grid.pings.filter((ping) => now - ping.at < 600);
    for (const ping of grid.pings) {
      const age = (now - ping.at) / 600;
      ctx.strokeStyle = `rgba(255, 255, 255, ${0.8 * (1 - age)})`;
      ctx.beginPath();
      ctx.arc(x0 + ((ping.cell % 32) + 0.5) * cell, y0 + (Math.floor(ping.cell / 32) + 0.5) * cell, cell * (1 + 3 * age), 0, 2 * Math.PI);
      ctx.stroke();
    }
    if (grid.outcomes && revealed < 1024) this.request();
  }
}

// Rotates v about a unit axis by angle (radians), by Rodrigues' formula.
function rotate(v, axis, angle) {
  if (!Number.isFinite(angle) || angle === 0) return v;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = axis;
  const kv = cross(k, v);
  const kd = dot(k, v) * (1 - c);
  return norm([0, 1, 2].map((i) => v[i] * c + kv[i] * s + k[i] * kd));
}
