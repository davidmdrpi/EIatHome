// Small canvas chart helpers. Colors come from CSS custom properties, so light and dark
// themes each use their own validated steps. Every chart gets a hover tooltip.

const tooltip = document.getElementById('tooltip');

export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const v = hex.replace('#', '');
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
}

const lerp = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));

function rampLookup(stops) {
  const rgb = stops.map(hexToRgb);
  return (f) => {
    const x = Math.min(1, Math.max(0, f)) * (rgb.length - 1);
    const i = Math.min(rgb.length - 2, Math.floor(x));
    return lerp(rgb[i], rgb[i + 1], x - i);
  };
}

function prepare(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = Number(canvas.dataset.height || 160);
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  return { ctx, w, h };
}

function attachHover(canvas, handler) {
  canvas._hover = handler;
  if (canvas._hoverBound) return;
  canvas._hoverBound = true;
  canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect();
    const html = canvas._hover?.(event.clientX - rect.left, event.clientY - rect.top);
    if (!html) return hideTooltip();
    tooltip.innerHTML = html;
    tooltip.hidden = false;
    const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
  });
  canvas.addEventListener('mouseleave', hideTooltip);
}

function hideTooltip() {
  tooltip.hidden = true;
}

export function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

export const fmt = (v, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '–');

// ---------- heatmap ----------

// kind: 'sequential' (magnitude, one hue) or 'diverging' (sign, blue <-> red, gray midpoint).
// rowLabel names the vertical axis; firstRow is the number of the bottom row.
export function heatmap(canvas, values, rows, cols, { kind = 'sequential', range, label = 'value', rowLabel = 'frame', firstRow = 2 } = {}) {
  const { ctx, w, h } = prepare(canvas);
  const pad = { l: 34, r: 6, t: 6, b: 22 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const [lo, hi] = range;
  const color =
    kind === 'diverging'
      ? rampLookup([token('--div-neg'), token('--div-mid'), token('--div-pos')])
      : rampLookup(Array.from({ length: 7 }, (_, i) => token(`--seq-${i}`)));

  const image = new ImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = values[r * cols + c];
      const [red, green, blue] = color((v - lo) / (hi - lo));
      const o = ((rows - 1 - r) * cols + c) * 4; // time runs upward
      image.data.set([red, green, blue, 255], o);
    }
  }
  const off = new OffscreenCanvas(cols, rows);
  off.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, pad.l, pad.t, pw, ph);

  ctx.fillStyle = token('--ink-muted');
  ctx.textAlign = 'center';
  ctx.fillText('channel', pad.l + pw / 2, h - 6);
  ctx.textAlign = 'right';
  ctx.fillText(String(firstRow + rows - 1), pad.l - 4, pad.t + 9);
  ctx.fillText(String(firstRow), pad.l - 4, pad.t + ph);
  ctx.save();
  ctx.translate(10, pad.t + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText(rowLabel, 0, 0);
  ctx.restore();

  attachHover(canvas, (x, y) => {
    if (x < pad.l || x > pad.l + pw || y < pad.t || y > pad.t + ph) return null;
    const c = Math.floor(((x - pad.l) / pw) * cols);
    const r = rows - 1 - Math.floor(((y - pad.t) / ph) * rows);
    return `${rowLabel} ${r + firstRow}, channel ${c}<br><b>${label} ${fmt(values[r * cols + c])}</b>`;
  });
}

// ---------- line chart ----------

// series: [{ name, color (CSS token name), points: [[x, y], ...], hollow?: (i) => bool,
//   band?: [[x, low, high], ...] (an interval, shaded), describe?: (i) => tooltip text for point i }]
// xTicks: tick positions, in place of the automatic ones.
export function lineChart(canvas, series, { xLabel = '', yLabel = '', logX = false, logY = false, yMin, yMax, markers = false, xTicks: fixedTicks, xFormat = (v) => fmt(v, 0), yFormat = (v) => fmt(v, 2) } = {}) {
  const { ctx, w, h } = prepare(canvas);
  const pad = { l: 44, r: 12, t: 22, b: 30 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const tx = logX ? Math.log10 : (v) => v;
  const ty = logY ? Math.log10 : (v) => v;
  const all = series.flatMap((s) => s.points).filter(([x, y]) => Number.isFinite(tx(x)) && Number.isFinite(ty(y)));
  if (!all.length) return;
  let [x0, x1] = [Math.min(...all.map((p) => tx(p[0]))), Math.max(...all.map((p) => tx(p[0])))];
  let [y0, y1] = [yMin ?? Math.min(...all.map((p) => ty(p[1]))), yMax ?? Math.max(...all.map((p) => ty(p[1])))];
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;
  const sx = (v) => pad.l + ((tx(v) - x0) / (x1 - x0)) * pw;
  const sy = (v) => pad.t + ph - ((ty(v) - y0) / (y1 - y0)) * ph;

  // grid and axes: solid hairlines
  ctx.lineWidth = 1;
  const powers = (a, b) => {
    const out = [];
    for (let e = Math.ceil(a - 1e-9); e <= Math.floor(b + 1e-9); e++) out.push(10 ** e);
    return out.length ? out : [10 ** a];
  };
  const yTicks = logY ? powers(y0, y1) : niceTicks(y0, y1);
  const xTicks = fixedTicks ?? (logX ? powers(x0, x1) : niceTicks(x0, x1, 5));
  ctx.strokeStyle = token('--grid');
  ctx.fillStyle = token('--ink-muted');
  ctx.textAlign = 'right';
  for (const v of yTicks) {
    const y = Math.round(sy(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillText(logY ? (v >= 1 ? String(v) : v.toExponential(0)) : yFormat(v), pad.l - 6, y + 4);
  }
  ctx.textAlign = 'center';
  for (const v of xTicks) ctx.fillText(xFormat(v), sx(v), pad.t + ph + 14);
  ctx.strokeStyle = token('--axis');
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ph + 0.5);
  ctx.lineTo(pad.l + pw, pad.t + ph + 0.5);
  ctx.stroke();
  ctx.fillText(xLabel, pad.l + pw / 2, h - 4);
  ctx.textAlign = 'left';
  ctx.fillText(yLabel, pad.l, 12);

  for (const s of series.filter((s) => s.band?.length)) {
    ctx.save();
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = token(s.color);
    ctx.beginPath();
    s.band.forEach(([x, low], i) => (i ? ctx.lineTo(sx(x), sy(low)) : ctx.moveTo(sx(x), sy(low))));
    for (const [x, , high] of [...s.band].reverse()) ctx.lineTo(sx(x), sy(high));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  for (const s of series) {
    const stroke = token(s.color);
    const pts = s.points.filter(([x, y]) => Number.isFinite(tx(x)) && Number.isFinite(ty(y)));
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
    ctx.stroke();
    if (markers) {
      pts.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = token('--surface');
        ctx.fill();
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), 4, 0, 2 * Math.PI);
        if (s.hollow?.(i)) {
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(sx(x), sy(y), 2, 0, 2 * Math.PI);
          ctx.fillStyle = token('--surface');
          ctx.fill();
        } else {
          ctx.fillStyle = stroke;
          ctx.fill();
        }
      });
    }
  }

  attachHover(canvas, (x, y) => {
    if (x < pad.l - 4 || x > pad.l + pw + 4 || y < pad.t - 4 || y > pad.t + ph + 4) return null;
    const rows = series
      .map((s) => {
        let best = -1;
        s.points.forEach((p, i) => {
          if (best < 0 || Math.abs(sx(p[0]) - x) < Math.abs(sx(s.points[best][0]) - x)) best = i;
        });
        return best >= 0 && `<span class="key" style="background:${token(s.color)}"></span>${s.name}: <b>${s.describe ? s.describe(best) : yFormat(s.points[best][1])}</b>`;
      })
      .filter(Boolean);
    let nearest = null;
    for (const p of series[0].points) if (!nearest || Math.abs(sx(p[0]) - x) < Math.abs(sx(nearest[0]) - x)) nearest = p;
    return nearest ? `${xLabel} ${xFormat(nearest[0])}<br>${rows.join('<br>')}` : null;
  });
}

// ---------- scatter (one panel of a small-multiples grid) ----------

export function scatterPanel(canvas, points, families, highlight, bounds, describe) {
  const { ctx, w, h } = prepare(canvas);
  const pad = 6;
  const [xMin, xMax, yMin, yMax] = bounds;
  const sx = (v) => pad + ((v - xMin) / (xMax - xMin || 1)) * (w - 2 * pad);
  const sy = (v) => h - pad - ((v - yMin) / (yMax - yMin || 1)) * (h - 2 * pad);
  ctx.strokeStyle = token('--grid');
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  ctx.fillStyle = token('--context');
  families.forEach((f, i) => {
    if (f === highlight) return;
    ctx.beginPath();
    ctx.arc(sx(points[2 * i]), sy(points[2 * i + 1]), 2, 0, 2 * Math.PI);
    ctx.fill();
  });
  families.forEach((f, i) => {
    if (f !== highlight) return;
    ctx.beginPath();
    ctx.arc(sx(points[2 * i]), sy(points[2 * i + 1]), 4, 0, 2 * Math.PI);
    ctx.fillStyle = token('--surface');
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx(points[2 * i]), sy(points[2 * i + 1]), 3, 0, 2 * Math.PI);
    ctx.fillStyle = token('--series-1');
    ctx.fill();
  });

  attachHover(canvas, (x, y) => {
    let best = -1;
    let bestD = 64;
    families.forEach((f, i) => {
      if (f !== highlight) return;
      const d = (sx(points[2 * i]) - x) ** 2 + (sy(points[2 * i + 1]) - y) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    return best >= 0 ? describe(best) : null;
  });
}
