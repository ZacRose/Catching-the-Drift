// app.js - the Drift Explorer page: controls, map, animation and results. The maths is in model.js;
// the data in data/app_data.js (src/app/export_app_data.py).
//
// Flow: pick a start (map click or typed position), then a timeframe. Choosing the timeframe runs the
// drift once; after the first run, changing the start or the timeframe runs it again. The results
// panel slides in beside the map when the drift has finished. "Loop" replays it until switched off.
// "Groundings" switches to the model in which material can also run aground, shown along the coast.
(function () {
  "use strict";

  const D = window.DRIFT_DATA;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  if (!D || !window.DriftModel || !D.intervals || !D.intervals.shore) {
    document.body.classList.add("no-data");
    $("#fatal").hidden = false;
    return;
  }

  const { Model, bearingDeg, compass } = window.DriftModel;
  const MODELS = { water: new Model(D, "water"), shore: new Model(D, "shore") };
  let M = MODELS.water; // the model in use; both share the grid
  const TAU = M.tau;
  const LEVELS = [0.5, 0.9];
  const REDUCE_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LOOP_PAUSE_MS = 1200;
  const FINE = 10; // contour raster: points per cell side
  const TIME_CHIPS = [[1, "15 days"], [2, "1 month"], [6, "3 months"], [12, "6 months"], [24, "1 year"], [72, "3 years"]]
    .filter(([s]) => s <= M.maxSteps);
  // Chance per cell, most likely first; the map and the legend share these bands.
  const HEAT_BINS = [
    { min: 0.1, label: "10%+", rgba: [255, 240, 190, 0.95] },
    { min: 0.01, label: "1–10%", rgba: [250, 204, 21, 0.88] },
    { min: 0.001, label: "0.1–1%", rgba: [45, 212, 191, 0.72] },
    { min: 0.0001, label: "0.01–0.1%", rgba: [14, 116, 144, 0.62] },
  ];
  // Chance it has run aground on a stretch of coast: the same bands in pink, drawn along the coastline.
  const SHORE_BINS = [
    { min: 0.1, rgba: [253, 224, 239, 1] },
    { min: 0.01, rgba: [244, 114, 182, 1] },
    { min: 0.001, rgba: [219, 39, 119, 1] },
    { min: 0.0001, rgba: [131, 24, 67, 1] },
  ];
  // Data flag: drifters that started a 15-day move in a cell (N = 100 is decision 3; under about 10,
  // bootstrap.py calls the intervals unreliable). The coverage layer uses the same colours.
  const FLAGS = [
    { min: 100, cls: "good", label: "Well sampled", short: "well sampled", range: "100+", rgba: [52, 211, 153, 0.5] },
    { min: 10, cls: "limited", label: "Limited data", short: "limited", range: "10–99", rgba: [251, 191, 36, 0.45] },
    { min: 1, cls: "thin", label: "Very thin data", short: "very thin", range: "under 10", rgba: [251, 113, 133, 0.45] },
  ];
  const flagOf = (n) => FLAGS.find((f) => n >= f.min) || null;

  // ---------- formatting ----------
  const num = (v) => (v >= 99.95 ? "100" : v >= 1 ? v.toFixed(1) : v < 0.001 ? "<0.001" : `${+v.toPrecision(2)}`);
  const fmtPct = (v) => (isFinite(v) ? `${num(v)}%` : "–");
  const km10 = (v) => (Math.round(v / 10) * 10).toLocaleString("en");
  const fmtKm = (v) => `${km10(v)} km`;
  const fmtLat = (v) => `${Math.abs(v).toFixed(1)}°${v < 0 ? "S" : v > 0 ? "N" : ""}`;
  const fmtLon = (v) => `${Math.abs(v).toFixed(1)}°${v < 0 ? "W" : v > 0 ? "E" : ""}`;
  const fmtDeg = (v, axis) => `${Math.abs(v)}°${v === 0 ? "" : axis === "lat" ? (v < 0 ? "S" : "N") : (v < 0 ? "W" : "E")}`;
  const half = (v) => { const r = Math.round(v * 2) / 2; return r % 1 === 0 ? `${r}` : r.toFixed(1); };
  function humanDays(d) {
    if (d < 30) return `${d} days`;
    if (d < 350) { const m = half(d / 30.44); return `about ${m} month${m === "1" ? "" : "s"}`; }
    const y = half(d / 365.25);
    return `about ${y} year${y === "1" ? "" : "s"}`;
  }
  const boxText = (b) => `${fmtDeg(b.lat_min, "lat")}–${fmtDeg(b.lat_max, "lat")}, ${fmtDeg(b.lon_min, "lon")}–${fmtDeg(b.lon_max, "lon")}`;
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const css = (c, a = c[3]) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // ---------- results rows ----------
  const QUANT = {
    centre_km: { label: "Distance travelled", desc: "Start to the centre (◆)", unit: "km", color: "#38bdf8", highlight: "centre" },
    pct_left_box: { label: "Left the study area", desc: "Not followed past the box", unit: "%", color: "#fb7185", highlight: "box" },
    pct_aground: { label: "Ran aground", desc: "On any coast in the box (pink)", unit: "%", color: "#f472b6", highlight: "shore" },
    pct_in_benguela: { label: "In the Benguela zone", desc: boxText(M.benguela), unit: "%", color: "#2dd4bf", highlight: "benguela" },
    pct_in_coastal_zone: { label: "Near the Benguela coast", desc: "Within ~200 km of the shore", unit: "%", color: "#fbbf24", highlight: "coast" },
  };
  const rowsFor = (ground) => ["centre_km", "pct_left_box", ...(ground ? ["pct_aground"] : []), "pct_in_benguela", "pct_in_coastal_zone"];

  const coastStart = () => ({ kind: "coast", name: "Whole Benguela coast" });
  const cellStart = (cell) => ({ kind: "cell", cell, name: `${fmtLat(M.latC[cell])}, ${fmtLon(M.lonC[cell])}` });
  const startKey = (s) => (s.kind === "coast" ? "coast" : `cell:${s.cell}`);

  const state = {
    view: "explore",
    start: coastStart(),
    steps: 6,
    t: 0,
    playing: false,
    loop: false,
    layers: { areas: true, ground: false, cells: false, coverage: false },
    hover: -1,
    highlight: null,
  };
  let R = null; // the current run: {start, steps, dists, stats, areas, contours, finished}
  let loopTimer = 0, dirty = true;

  // ---------- map ----------
  const wrap = $("#mapWrap");
  const canvases = { base: $("#baseCanvas"), heat: $("#heatCanvas"), fx: $("#fxCanvas") };
  const ctx = Object.fromEntries(Object.entries(canvases).map(([k, c]) => [k, c.getContext("2d")]));
  const view = (() => {
    const g = M.region, pad = 3;
    return { lon0: g.lon_min - pad, lon1: g.lon_max + pad, lat0: g.lat_min - pad, lat1: g.lat_max + pad, kx: Math.cos(((g.lat_min + g.lat_max) / 2) * Math.PI / 180) };
  })();
  const map = { w: 0, h: 0, s: 1, ox: 0, oy: 0, dpr: 1, land: null, shorePaths: [] };
  const X = (lon) => map.ox + (lon - view.lon0) * view.kx * map.s;
  const Y = (lat) => map.oy + (view.lat1 - lat) * map.s;
  const lonAt = (x) => (x - map.ox) / (view.kx * map.s) + view.lon0;
  const latAt = (y) => view.lat1 - (y - map.oy) / map.s;
  const font = (w, px) => `${w} ${px}px ${getComputedStyle(document.body).fontFamily}`;
  const cellTex = document.createElement("canvas"); // one pixel per cell (grid view)
  cellTex.width = M.nLon; cellTex.height = M.nLat;
  const fineTex = document.createElement("canvas"); // FINE pixels per cell (contour view)
  fineTex.width = M.nLon * FINE; fineTex.height = M.nLat * FINE;
  const landLayer = document.createElement("canvas");

  // Coastline in short pieces, sorted by the stretch of coast (shore state) each belongs to: a piece
  // belongs to a shore cell if its middle lies in the cell or within 0.25 deg of it.
  const shorePieces = (() => {
    const cells = D.shore.cells, out = cells.map(() => []), pad = 0.25;
    const bounds = cells.map((c) => M.cellBounds(c));
    for (const ring of D.land) {
      for (let i = 1; i < ring.length; i++) {
        const [x0, y0] = ring[i - 1], [x1, y1] = ring[i];
        const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 0.2));
        for (let j = 0; j < n; j++) {
          const a = [lerp(x0, x1, j / n), lerp(y0, y1, j / n)], b = [lerp(x0, x1, (j + 1) / n), lerp(y0, y1, (j + 1) / n)];
          const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
          bounds.forEach((bd, k) => {
            if (mx >= bd.lon0 - pad && mx <= bd.lon1 + pad && my >= bd.lat0 - pad && my <= bd.lat1 + pad) out[k].push([a, b]);
          });
        }
      }
    }
    return out;
  })();

  function resize() {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    if (!w || !h) return;
    const ux = (view.lon1 - view.lon0) * view.kx, uy = view.lat1 - view.lat0;
    map.s = Math.min(w / ux, h / uy);
    map.ox = (w - ux * map.s) / 2; map.oy = (h - uy * map.s) / 2;
    map.w = w; map.h = h; map.dpr = Math.min(window.devicePixelRatio || 1, 2);
    for (const k of ["base", "heat", "fx"]) {
      canvases[k].width = Math.round(w * map.dpr); canvases[k].height = Math.round(h * map.dpr);
      ctx[k].setTransform(map.dpr, 0, 0, map.dpr, 0, 0);
    }
    landLayer.width = canvases.base.width; landLayer.height = canvases.base.height;
    map.land = new Path2D();
    for (const ring of D.land) {
      ring.forEach(([lon, lat], i) => (i ? map.land.lineTo(X(lon), Y(lat)) : map.land.moveTo(X(lon), Y(lat))));
      map.land.closePath();
    }
    map.shorePaths = shorePieces.map((pieces) => {
      const p = new Path2D();
      for (const [a, b] of pieces) { p.moveTo(X(a[0]), Y(a[1])); p.lineTo(X(b[0]), Y(b[1])); }
      return p;
    });
    if (R) R.contours = new Map();
    drawBase(); drawLandLayer(); dirty = true;
  }

  function drawBase() {
    const c = ctx.base, g = M.region;
    c.clearRect(0, 0, map.w, map.h);
    const sea = c.createLinearGradient(0, 0, 0, map.h);
    sea.addColorStop(0, "#0c3150"); sea.addColorStop(0.55, "#092640"); sea.addColorStop(1, "#06182b");
    c.fillStyle = sea; c.fillRect(0, 0, map.w, map.h);
    c.strokeStyle = "rgba(148,190,230,0.07)"; c.lineWidth = 1; c.fillStyle = "rgba(156,178,198,0.5)";
    c.font = font(400, Math.max(9, map.s * 0.85));
    for (let lon = Math.ceil(view.lon0 / 10) * 10; lon <= view.lon1; lon += 10) {
      c.beginPath(); c.moveTo(X(lon), Y(view.lat1)); c.lineTo(X(lon), Y(view.lat0)); c.stroke();
      c.textAlign = "center"; c.fillText(fmtDeg(lon, "lon"), X(lon), Y(view.lat0) - 5);
    }
    for (let lat = Math.ceil(view.lat0 / 10) * 10; lat <= view.lat1; lat += 10) {
      c.beginPath(); c.moveTo(X(view.lon0), Y(lat)); c.lineTo(X(view.lon1), Y(lat)); c.stroke();
      c.textAlign = "left"; c.fillText(fmtDeg(lat, "lat"), X(view.lon0) + 5, Y(lat) - 4);
    }
    // outside the study area is shaded: the model has no states there
    c.fillStyle = "rgba(2,8,16,0.42)";
    c.beginPath(); c.rect(0, 0, map.w, map.h); c.rect(X(g.lon_min), Y(g.lat_min), X(g.lon_max) - X(g.lon_min), Y(g.lat_max) - Y(g.lat_min));
    c.fill("evenodd");
  }

  function drawLandLayer() {
    const c = landLayer.getContext("2d"), g = M.region;
    c.setTransform(map.dpr, 0, 0, map.dpr, 0, 0);
    c.clearRect(0, 0, map.w, map.h);
    c.fillStyle = "#283743"; c.fill(map.land);
    c.strokeStyle = "rgba(186,214,235,0.4)"; c.lineWidth = 0.9; c.stroke(map.land);
    c.fillStyle = "rgba(205,222,236,0.28)"; c.textAlign = "center"; c.font = font(650, Math.max(9, map.s * 1.0));
    c.save(); c.letterSpacing = "0.3em";
    c.fillText("AFRICA", X(18), Y(-9)); c.fillText("SOUTH AMERICA", X(-53), Y(-12));
    c.restore();
    c.strokeStyle = "rgba(148,197,255,0.35)"; c.lineWidth = 1;
    c.strokeRect(X(g.lon_min), Y(g.lat_max), X(g.lon_max) - X(g.lon_min), Y(g.lat_min) - Y(g.lat_max));
  }

  // ---------- the continuous field behind the contours ----------
  // Each fine point mixes its four nearest cell centres bilinearly, on a log scale, using only cells that
  // aren't all land, so bands run up to the coast instead of fading towards land cells.
  const field = (() => {
    const nx = M.nLon * FINE, ny = M.nLat * FINE, idx = new Int32Array(nx * ny * 4), wt = new Float32Array(nx * ny * 4);
    for (let py = 0; py < ny; py++) {
      const v = (ny - py - 0.5) / FINE - 0.5, rv = Math.floor(v), fv = v - rv;
      const r0 = clamp(rv, 0, M.nLat - 1), r1 = clamp(rv + 1, 0, M.nLat - 1);
      for (let px = 0; px < nx; px++) {
        const u = (px + 0.5) / FINE - 0.5, ku = Math.floor(u), fu = u - ku;
        const k0 = clamp(ku, 0, M.nLon - 1), k1 = clamp(ku + 1, 0, M.nLon - 1), i = (py * nx + px) * 4;
        idx[i] = r0 * M.nLon + k0; wt[i] = (1 - fu) * (1 - fv);
        idx[i + 1] = r0 * M.nLon + k1; wt[i + 1] = fu * (1 - fv);
        idx[i + 2] = r1 * M.nLon + k0; wt[i + 2] = (1 - fu) * fv;
        idx[i + 3] = r1 * M.nLon + k1; wt[i + 3] = fu * fv;
        for (let j = i; j < i + 4; j++) if (M.allLand[idx[j]]) wt[j] = 0;
      }
    }
    return { nx, ny, idx, wt, logs: new Float32Array(M.nCells) };
  })();
  function fineValues(perCell, floor) { // log10 of a per-cell quantity at every fine point (NaN where only land)
    const { nx, ny, idx, wt, logs } = field, out = new Float32Array(nx * ny);
    for (let c = 0; c < M.nCells; c++) logs[c] = Math.log10(Math.max(perCell(c), floor));
    for (let p = 0, i = 0; p < out.length; p++, i += 4) {
      const w = wt[i] + wt[i + 1] + wt[i + 2] + wt[i + 3];
      out[p] = w > 0 ? (wt[i] * logs[idx[i]] + wt[i + 1] * logs[idx[i + 1]] + wt[i + 2] * logs[idx[i + 2]] + wt[i + 3] * logs[idx[i + 3]]) / w : NaN;
    }
    return smooth(out, nx, ny, Math.round(FINE * 0.35));
  }
  // Two passes of a box blur over the valid (non-NaN) points, renormalised so land doesn't drag the
  // field down: rounds off the corners that straight-line mixing leaves at cell centres.
  function smooth(vals, nx, ny, r) {
    let v = Float32Array.from(vals, (x) => (x === x ? x : 0)), m = Float32Array.from(vals, (x) => (x === x ? 1 : 0));
    const pass = (a, horizontal) => {
      const out = new Float32Array(a.length), n = horizontal ? nx : ny, lines = horizontal ? ny : nx;
      for (let l = 0; l < lines; l++) {
        const at = (k) => (horizontal ? l * nx + k : k * nx + l);
        let sum = 0;
        for (let k = -r; k <= r; k++) sum += a[at(clamp(k, 0, n - 1))];
        for (let k = 0; k < n; k++) {
          out[at(k)] = sum / (2 * r + 1);
          sum += a[at(clamp(k + r + 1, 0, n - 1))] - a[at(clamp(k - r, 0, n - 1))];
        }
      }
      return out;
    };
    for (let rep = 0; rep < 2; rep++) {
      v = pass(pass(v, true), false);
      m = pass(pass(m, true), false);
    }
    for (let p = 0; p < vals.length; p++) vals[p] = vals[p] === vals[p] && m[p] > 0 ? v[p] / m[p] : NaN;
    return vals;
  }
  // Contour of the fine field at a log level: marching squares through the fine points, with the pieces
  // joined into continuous lines (so dashes run along them) by the grid edge each crossing lies on.
  function contourPath(vals, level) {
    const { nx, ny } = field, g = M.region, H = nx * ny;
    const px = (i) => X(g.lon_min + ((i + 0.5) / FINE) * M.size), py = (j) => Y(g.lat_max - ((j + 0.5) / FINE) * M.size);
    const pts = new Map(), links = new Map();
    const hit = (key, i0, j0, v0, i1, j1, v1) => {
      if (!pts.has(key)) { const f = (level - v0) / (v1 - v0); pts.set(key, [px(i0 + (i1 - i0) * f), py(j0 + (j1 - j0) * f)]); }
      return key;
    };
    const link = (a, b) => { (links.get(a) || links.set(a, []).get(a)).push(b); (links.get(b) || links.set(b, []).get(b)).push(a); };
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const a = vals[j * nx + i], b = vals[j * nx + i + 1], c = vals[(j + 1) * nx + i + 1], d = vals[(j + 1) * nx + i];
        if (!(a === a && b === b && c === c && d === d)) continue; // NaN: land
        const k = [];
        if ((a > level) !== (b > level)) k.push(hit(j * nx + i, i, j, a, i + 1, j, b)); // top edge
        if ((b > level) !== (c > level)) k.push(hit(H + j * nx + i + 1, i + 1, j, b, i + 1, j + 1, c)); // right
        if ((c > level) !== (d > level)) k.push(hit((j + 1) * nx + i, i + 1, j + 1, c, i, j + 1, d)); // bottom
        if ((d > level) !== (a > level)) k.push(hit(H + j * nx + i, i, j + 1, d, i, j, a)); // left
        for (let q = 0; q + 1 < k.length; q += 2) link(k[q], k[q + 1]);
      }
    }
    const p = new Path2D(), seen = new Set();
    const walk = (from) => { // follow links from one end (or anywhere on a loop) until they run out
      let prev = -1, cur = from;
      p.moveTo(...pts.get(cur)); seen.add(cur);
      for (;;) {
        const next = (links.get(cur) || []).find((n) => n !== prev && !seen.has(n));
        if (next === undefined) {
          if ((links.get(cur) || []).includes(from) && cur !== from && prev !== from) p.closePath();
          return;
        }
        p.lineTo(...pts.get(next)); seen.add(next);
        prev = cur; cur = next;
      }
    };
    for (const [key, nb] of links) if (nb.length === 1 && !seen.has(key)) walk(key); // open lines first
    for (const key of links.keys()) if (!seen.has(key)) walk(key); // then loops
    return p;
  }

  const blend = new Float64Array(Math.max(MODELS.water.nStates, MODELS.shore.nStates));
  function currentDist() {
    if (!R) return null;
    const k = clamp(Math.floor(state.t), 0, R.steps), f = state.t - k;
    if (f < 1e-6 || k >= R.steps) return R.dists[k];
    const a = R.dists[k], b = R.dists[k + 1];
    for (let i = 0; i < a.length; i++) blend[i] = a[i] + (b[i] - a[i]) * f;
    return blend;
  }
  function drawHeat() {
    const c = ctx.heat, g = M.region, explore = state.view === "explore", cover = explore && state.layers.coverage;
    const dist = !explore || cover ? null : currentDist();
    c.clearRect(0, 0, map.w, map.h);
    if (cover || dist) {
      const binOf = cover ? (v) => { const f = flagOf(10 ** v + 1e-9); return f && f.rgba; }
        : (v) => { const b = HEAT_BINS.find((h) => v >= Math.log10(h.min)); return b && b.rgba; };
      let tex;
      if (state.layers.cells) { // one colour per cell
        tex = cellTex;
        const img = cellTex.getContext("2d").createImageData(M.nLon, M.nLat);
        for (let cell = 0; cell < M.nCells; cell++) {
          const v = cover ? (M.drifters[cell] > 0 ? Math.log10(M.drifters[cell]) : NaN) : Math.log10(dist[cell]);
          const col = v === v ? binOf(v) : null;
          if (!col) continue;
          const i = ((M.nLat - 1 - Math.floor(cell / M.nLon)) * M.nLon + (cell % M.nLon)) * 4;
          img.data[i] = col[0]; img.data[i + 1] = col[1]; img.data[i + 2] = col[2]; img.data[i + 3] = Math.round(col[3] * 255);
        }
        cellTex.getContext("2d").putImageData(img, 0, 0);
      } else { // continuous field in bands (filled contours)
        tex = fineTex;
        const vals = cover ? fineValues((cell) => M.drifters[cell], 0.1) : fineValues((cell) => dist[cell], 1e-9);
        const img = fineTex.getContext("2d").createImageData(field.nx, field.ny);
        for (let p = 0; p < vals.length; p++) {
          const v = vals[p], col = v === v ? binOf(v) : null;
          if (!col) continue;
          img.data[p * 4] = col[0]; img.data[p * 4 + 1] = col[1]; img.data[p * 4 + 2] = col[2]; img.data[p * 4 + 3] = Math.round(col[3] * 255);
        }
        fineTex.getContext("2d").putImageData(img, 0, 0);
      }
      c.save();
      c.imageSmoothingEnabled = !state.layers.cells; c.imageSmoothingQuality = "high";
      c.drawImage(tex, X(g.lon_min), Y(g.lat_max), X(g.lon_max) - X(g.lon_min), Y(g.lat_min) - Y(g.lat_max));
      c.restore();
    }
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(landLayer, 0, 0); c.restore();
  }

  function cellRect(cell) {
    const b = M.cellBounds(cell);
    return [X(b.lon0), Y(b.lat1), X(b.lon1) - X(b.lon0), Y(b.lat0) - Y(b.lat1)];
  }
  function outlinePath(mask) { // boundary edges of a set of cells
    const p = new Path2D();
    for (let cell = 0; cell < M.nCells; cell++) {
      if (!mask[cell]) continue;
      const r = Math.floor(cell / M.nLon), k = cell % M.nLon, b = M.cellBounds(cell);
      const x0 = X(b.lon0), x1 = X(b.lon1), y0 = Y(b.lat1), y1 = Y(b.lat0);
      if (r === M.nLat - 1 || !mask[cell + M.nLon]) { p.moveTo(x0, y0); p.lineTo(x1, y0); }
      if (r === 0 || !mask[cell - M.nLon]) { p.moveTo(x0, y1); p.lineTo(x1, y1); }
      if (k === 0 || !mask[cell - 1]) { p.moveTo(x0, y0); p.lineTo(x0, y1); }
      if (k === M.nLon - 1 || !mask[cell + 1]) { p.moveTo(x1, y0); p.lineTo(x1, y1); }
    }
    return p;
  }
  // The 50% / 90% outlines at a whole step: the cells themselves in the grid view, otherwise the contour
  // of the smooth field at the level between the last cell in and the first cell out.
  function areasAt(step) {
    if (!R.areas.has(step)) R.areas.set(step, M.likelyAreas(R.dists[step], LEVELS));
    const areas = R.areas.get(step), key = `${step}:${state.layers.cells ? "cells" : "smooth"}`;
    if (!R.contours.has(key)) {
      let paths;
      if (state.layers.cells) paths = areas.map((a) => outlinePath(a.mask));
      else {
        const vals = fineValues((cell) => R.dists[step][cell], 1e-9);
        paths = areas.map((a) => contourPath(vals, (Math.log10(a.pMin) + Math.log10(Math.max(a.pNext, a.pMin * 1e-3))) / 2));
      }
      R.contours.set(key, paths);
    }
    return R.contours.get(key);
  }

  function drawFx() {
    const c = ctx.fx, B = M.benguela, g = M.region, hl = state.highlight;
    c.clearRect(0, 0, map.w, map.h);

    if (hl === "benguela") {
      const x = X(B.lon_min), y = Y(B.lat_max), w = X(B.lon_max) - x, h = Y(B.lat_min) - y;
      c.save();
      c.fillStyle = "rgba(45,212,191,0.08)"; c.fillRect(x, y, w, h);
      c.setLineDash([6, 5]); c.lineWidth = 2; c.strokeStyle = "rgba(45,212,191,1)"; c.strokeRect(x, y, w, h);
      c.fillStyle = "rgba(45,212,191,0.95)"; c.font = font(650, Math.max(10, map.s * 0.8)); c.textAlign = "left";
      c.fillText("BENGUELA ZONE", x + 4, y - 6);
      c.restore();
    }
    if (hl === "coast") {
      c.save(); c.fillStyle = "rgba(251,191,36,0.2)"; c.strokeStyle = "rgba(251,191,36,0.95)"; c.lineWidth = 1.4;
      for (const cell of D.coastal_cells) { const [x, y, w, h] = cellRect(cell); c.fillRect(x, y, w, h); c.strokeRect(x, y, w, h); }
      c.restore();
    }
    if (hl === "box") {
      c.save(); c.strokeStyle = "rgba(251,113,133,0.95)"; c.lineWidth = 2.5; c.shadowColor = "rgba(251,113,133,0.7)"; c.shadowBlur = 8;
      c.strokeRect(X(g.lon_min), Y(g.lat_max), X(g.lon_max) - X(g.lon_min), Y(g.lat_min) - Y(g.lat_max));
      c.restore();
    }

    if (state.view !== "explore") return; // the Seasons view shows only the zone its chart is about
    const show = R && !state.layers.coverage && state.t > 0;
    // where it has run aground: the coastline coloured by the chance, stretch by stretch
    if (show && state.layers.ground) {
      const dist = currentDist();
      c.save(); c.lineCap = "round"; c.lineJoin = "round";
      M.shoreCells.forEach((cell, k) => {
        const p = dist[M.SHORE0 + k], bin = SHORE_BINS.find((b) => p >= b.min);
        if (!bin && hl !== "shore") return;
        const path = map.shorePaths[k];
        c.lineWidth = hl === "shore" ? 9 : 7; c.strokeStyle = "rgba(4,14,26,0.8)"; c.stroke(path);
        c.lineWidth = hl === "shore" ? 5.5 : 4; c.strokeStyle = bin ? css(bin.rgba) : "rgba(244,114,182,0.35)"; c.stroke(path);
      });
      c.restore();
    }
    if (show) {
      const step = clamp(Math.round(state.t), 0, R.steps);
      if (state.layers.areas && step > 0) {
        const [p50, p90] = areasAt(step);
        // white over a dark edge, so both lines read on every colour band
        c.save(); c.lineJoin = "round"; c.lineCap = "round";
        for (const [path, dash] of [[p90, [6, 4]], [p50, []]]) {
          c.setLineDash(dash);
          c.lineWidth = 4; c.strokeStyle = "rgba(4,14,26,0.75)"; c.stroke(path);
          c.lineWidth = dash.length ? 1.8 : 2.2; c.strokeStyle = "#ffffff"; c.stroke(path);
        }
        c.restore();
      }
      // the centre of the material in the water and the path it has taken
      const k = clamp(Math.floor(state.t), 0, R.steps), f = state.t - k;
      c.save();
      c.beginPath();
      for (let s = 0; s <= k; s++) { const st = R.stats[s]; s ? c.lineTo(X(st.centre_lon), Y(st.centre_lat)) : c.moveTo(X(st.centre_lon), Y(st.centre_lat)); }
      let cx = X(R.stats[k].centre_lon), cy = Y(R.stats[k].centre_lat);
      if (k < R.steps && f > 0) {
        cx = lerp(cx, X(R.stats[k + 1].centre_lon), f); cy = lerp(cy, Y(R.stats[k + 1].centre_lat), f);
        c.lineTo(cx, cy);
      }
      c.setLineDash([2, 4]); c.lineWidth = 1.6; c.strokeStyle = "rgba(255,255,255,0.8)"; c.stroke();
      c.setLineDash([]);
      const r = hl === "centre" ? 7 : 5;
      if (hl === "centre") { c.beginPath(); c.arc(cx, cy, 14, 0, Math.PI * 2); c.strokeStyle = "rgba(56,189,248,0.95)"; c.lineWidth = 2; c.stroke(); }
      c.translate(cx, cy); c.rotate(Math.PI / 4);
      c.shadowColor = "rgba(0,0,0,0.6)"; c.shadowBlur = 6; c.fillStyle = "#fff"; c.fillRect(-r, -r, 2 * r, 2 * r);
      c.shadowBlur = 0; c.strokeStyle = "#0a2540"; c.lineWidth = 1.5; c.strokeRect(-r, -r, 2 * r, 2 * r);
      c.restore();
    }

    // the start
    c.save();
    if (state.start.kind === "coast") {
      c.strokeStyle = "rgba(251,191,36,0.9)"; c.lineWidth = 1.3;
      for (const cell of M.releaseCells) { const [x, y, w, h] = cellRect(cell); c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
    }
    const o = M.startPoint(state.start), sx = X(o.lon), sy = Y(o.lat);
    c.beginPath(); c.arc(sx, sy, 11, 0, Math.PI * 2); c.strokeStyle = "rgba(251,191,36,0.45)"; c.lineWidth = 2; c.stroke();
    c.beginPath(); c.arc(sx, sy, 5.5, 0, Math.PI * 2); c.fillStyle = "#fbbf24"; c.fill();
    c.lineWidth = 2; c.strokeStyle = "#1a1200"; c.stroke();
    c.restore();

    if (state.hover >= 0) {
      const [x, y, w, h] = cellRect(state.hover);
      c.strokeStyle = "rgba(255,255,255,0.9)"; c.lineWidth = 1.5; c.strokeRect(x, y, w, h);
    }
  }

  // ---------- legend (a strip under the map) ----------
  const keyLine = (dash, colour, width) => `<svg viewBox="0 0 26 12" aria-hidden="true"><line x1="1" y1="6" x2="25" y2="6" stroke="${colour}" stroke-width="${width}" ${dash ? 'stroke-dasharray="4 3"' : ""}/></svg>`;
  function renderLegend() {
    const L = $("#legend");
    if (state.layers.coverage) {
      L.innerHTML = `<div class="lg-group"><span class="lg-title">Drifters that started a 15-day move in each 2° square</span>
        <span class="lg-bins">${[...FLAGS].reverse().map((f) => `<span><i style="background:${css(f.rgba, 0.9)}"></i>${f.range} ${f.short}</span>`).join("")}</span></div>`;
      return;
    }
    const bins = [...HEAT_BINS].reverse().map((b) => `<span><i style="background:${css(b.rgba)}"></i>${b.label}</span>`).join("");
    const shore = state.layers.ground ? `<div class="lg-group"><span class="lg-title">Ran aground on that stretch of coast</span>
        <span class="lg-bins">${[...SHORE_BINS].reverse().map((b, i) => `<span><i class="line" style="background:${css(b.rgba)}"></i>${[...HEAT_BINS].reverse()[i].label}</span>`).join("")}</span></div>` : "";
    const keys = R ? `<div class="lg-group lg-keys">
        ${state.layers.areas ? `<span>${keyLine(false, "#ffffff", 2.4)}50% chance it's inside</span><span>${keyLine(true, "#ffffff", 1.8)}90% chance it's inside</span>` : ""}
        <span><svg viewBox="0 0 26 12" aria-hidden="true"><rect x="9" y="2" width="8" height="8" transform="rotate(45 13 6)" fill="#fff" stroke="#0a2540"/></svg>Centre and its path</span>
      </div>` : "";
    L.innerHTML = `<div class="lg-group"><span class="lg-title">${state.layers.ground ? "Still in the water: chance" : "Chance it's"} in each 2° square (about 200 km across)</span>
      <span class="lg-bins">${bins}</span></div>${shore}${keys}`;
  }

  // ---------- start card ----------
  function renderStart() {
    const s = state.start;
    let where, flag;
    if (s.kind === "coast") {
      const n = M.releaseCells.map((c) => M.drifters[c]).sort((a, b) => a - b);
      flag = flagOf(n[Math.floor(n.length / 2)]);
      where = `${n.length} coastal squares, ${fmtDeg(M.benguela.lat_min, "lat")}–${fmtDeg(M.benguela.lat_max, "lat")} · ${n[0]}–${n[n.length - 1]} drifters each, ${n.filter((v) => v < 10).length} under 10`;
    } else {
      const b = M.cellBounds(s.cell), n = M.drifters[s.cell];
      flag = flagOf(n);
      where = `2° square ${fmtDeg(b.lat0, "lat")}–${fmtDeg(b.lat1, "lat")}, ${fmtDeg(b.lon0, "lon")}–${fmtDeg(b.lon1, "lon")} · ${n} drifter${n === 1 ? "" : "s"}`;
    }
    const coastLink = s.kind !== "coast" ? `<button type="button" class="ss-coast" id="useCoast">Release along the whole Benguela coast instead</button>` : "";
    $("#startSummary").innerHTML = `
      <div class="ss-top"><span class="ss-name">${s.name}</span><span class="badge flag-${flag.cls}">${flag.label}</span></div>
      <div class="ss-where">${where}</div>${coastLink}
      <div class="flag-key"><span>Data flag, by drifters that started here:</span>
        <ul>${FLAGS.map((f) => `<li class="${f.cls}${f === flag ? " on" : ""}"><i></i>${f.range} ${f.short}</li>`).join("")}</ul></div>`;
    const btn = $("#useCoast");
    if (btn) btn.onclick = () => setStart(coastStart());
  }

  // ---------- results ----------
  function spark(q, step) {
    const W = 56, H = 28, n = R.steps;
    const est = R.stats.map((s) => s[q]);
    const ivs = est.map((_, s) => M.interval(R.start, q, s));
    const top = Math.max(...est, ...ivs.map((v) => (v ? v[1] : 0))) || 1;
    const x = (s) => (W * s) / n, y = (v) => H - 2 - ((H - 4) * v) / top;
    const line = est.map((v, s) => `${s ? "L" : "M"}${x(s).toFixed(1)},${y(v).toFixed(1)}`).join("");
    const up = est.map((v, s) => `${s ? "L" : "M"}${x(s).toFixed(1)},${y(ivs[s] ? ivs[s][1] : v).toFixed(1)}`).join("");
    const down = est.map((v, s) => [x(s), y(ivs[s] ? ivs[s][0] : v)]).reverse().map(([a, b]) => `L${a.toFixed(1)},${b.toFixed(1)}`).join("");
    const c = QUANT[q].color;
    return `<svg class="ch-spark" viewBox="0 0 ${W} ${H}" aria-hidden="true"><path class="band" d="${up}${down}Z" fill="${c}"/>
      <path class="line" d="${line}" stroke="${c}"/><circle cx="${x(step)}" cy="${y(est[step])}" r="3" fill="${c}" stroke="#0b1f33" stroke-width="1.5"/></svg>`;
  }
  function renderResults(step) {
    const st = R.stats[step], days = step * TAU, o = M.startPoint(R.start);
    const dir = st.centre_km > 30 ? compass(bearingDeg(o.lat, o.lon, st.centre_lat, st.centre_lon)) : null;
    const rows = rowsFor(R.ground).map((q) => {
      const Q = QUANT[q], iv = M.interval(R.start, q, step), v = st[q];
      const value = Q.unit === "km" ? fmtKm(v) : fmtPct(v);
      const range = !iv ? "" : Q.unit === "km" ? `${km10(iv[0])}–${km10(iv[1])} km` : iv[1] < 0.001 ? "under 0.001%" : `${num(iv[0])}–${num(iv[1])}%`;
      return `<li data-hl="${Q.highlight}"><div><div class="ch-label"><i style="background:${Q.color}"></i>${Q.label}</div><div class="ch-desc">${Q.desc}</div></div>
        ${spark(q, step)}<div class="ch-num"><div class="ch-val">${value}</div><div class="ch-rng">${range}</div></div></li>`;
    }).join("");
    $("#results").innerHTML = `<p class="eyebrow">After ${days} days${days >= 30 ? ` · ${humanDays(days)}` : ""}</p>
      <h3>${R.ground ? "Still afloat, most" : "Most"} likely around ${fmtLat(st.centre_lat)}, ${fmtLon(st.centre_lon)}</h3>
      <p class="where">${dir ? `${cap(dir)} of the start` : "Close to the start"}, marked ◆ on the map.</p>
      <ul class="chances">${rows}</ul>
      <p class="foot">Small charts: release to day ${R.steps * TAU}. Grey numbers: 95% ranges, how much each answer could shift with a
        different sample of drifters. Point at a row to see it on the map.</p>`;
    $$("#results .chances li").forEach((li) => {
      li.onmouseenter = () => { state.highlight = li.dataset.hl; dirty = true; };
      li.onmouseleave = () => { state.highlight = null; dirty = true; };
    });
  }
  // Bring the panel into view: on wide screens by scrolling the side panel alone, so the map stays put.
  function reveal(el) {
    const side = $(".side"), behavior = REDUCE_MOTION ? "auto" : "smooth";
    if (getComputedStyle(side).overflowY === "auto") {
      const over = el.getBoundingClientRect().bottom - side.getBoundingClientRect().bottom;
      if (over > 0) side.scrollBy({ top: over + 4, behavior });
    } else el.scrollIntoView({ behavior, block: "nearest" });
  }
  function fitSide() { // the side panel fills the window below the header
    const side = $(".side");
    side.style.maxHeight = window.innerWidth > 1100 ? `${window.innerHeight - side.getBoundingClientRect().top - window.scrollY - 12}px` : "";
  }
  let shownStep = -1;
  function syncResults(force) {
    const el = $("#results");
    const step = R ? (state.playing ? R.steps : clamp(Math.round(state.t), 0, R.steps)) : 0;
    const visible = !!R && state.view === "explore" && step > 0 && (R.finished || !state.playing);
    if (!visible) {
      if (!el.hidden) { el.classList.remove("show"); el.hidden = true; state.highlight = null; }
      shownStep = -1;
      return;
    }
    if (step !== shownStep || force) { renderResults(step); shownStep = step; }
    if (el.hidden) {
      el.hidden = false; void el.offsetWidth; el.classList.add("show");
      if (!R.revealed) reveal(el);
      R.revealed = true;
    }
  }

  // ---------- time ----------
  const stepMs = (n) => clamp(5200 / n, 110, 1400);
  function stepTicks(n, most = 6) { // whole steps at an even spacing of 15 days to a year
    const every = [1, 2, 6, 12, 24].find((e) => n / e <= most - 1) || 24;
    const ticks = [];
    for (let s = 0; s <= n; s += every) ticks.push(s);
    if (n - ticks[ticks.length - 1] >= every / 2) ticks.push(n);
    return ticks;
  }
  function updateTimeUI() {
    const has = !!R, ended = has && !state.playing && state.t >= R.steps;
    $("#dayCounter").hidden = !has;
    $("#dayNum").textContent = Math.round(state.t * TAU);
    const scrub = $("#scrub"), pb = $("#playBtn");
    scrub.disabled = pb.disabled = !has;
    if (document.activeElement !== scrub) scrub.value = state.t;
    scrub.style.setProperty("--fill", `${has ? (100 * state.t) / R.steps : 0}%`);
    pb.classList.toggle("playing", state.playing);
    pb.classList.toggle("ended", ended);
    pb.setAttribute("aria-label", state.playing ? "Pause" : ended ? "Replay" : "Play");
  }
  function renderScrubTicks() {
    const n = R.steps;
    $("#scrubTicks").innerHTML = stepTicks(n, map.w < 560 ? 3 : 6).map((s) => `<span style="left:${(100 * s) / n}%">${s ? `${s * TAU} d` : "Release"}</span>`).join("");
    $("#scrub").max = n;
    $("#dayOf").textContent = `of ${n * TAU}`;
  }
  function play(fromStart) {
    clearTimeout(loopTimer);
    if (fromStart) state.t = 0;
    state.playing = !REDUCE_MOTION;
    if (REDUCE_MOTION) { state.t = R.steps; R.finished = true; }
    dirty = true;
  }
  function onPlayEnd() {
    state.playing = false;
    R.finished = true;
    if (state.loop) loopTimer = setTimeout(() => { if (R && state.loop && !state.playing && state.t >= R.steps) play(true); }, LOOP_PAUSE_MS);
  }

  // ---------- running ----------
  function compute() {
    M = state.layers.ground ? MODELS.shore : MODELS.water;
    const start = state.start, dists = M.propagate(M.startVector(start), state.steps);
    return { start, steps: state.steps, ground: state.layers.ground, dists, stats: dists.map((d) => M.stats(d, start)),
      areas: new Map(), contours: new Map(), finished: false, revealed: false };
  }
  function run() {
    R = compute();
    shownStep = -1;
    $("#timeStep").classList.remove("await");
    $("#mapHint").textContent = "";
    renderScrubTicks(); renderLegend();
    play(true);
    syncResults(); // hide the last run's results straight away
    writeHash();
  }
  // Switching models keeps the run where it is (same start, timeframe and moment), without replaying.
  function rerunInPlace() {
    if (!R) { M = state.layers.ground ? MODELS.shore : MODELS.water; return; }
    const keep = { t: Math.min(state.t, state.steps), finished: R.finished, revealed: R.revealed };
    R = Object.assign(compute(), { finished: keep.finished, revealed: keep.revealed });
    state.t = keep.t;
    syncResults(true);
    writeHash();
  }

  function setStart(start) {
    state.start = start;
    renderStart();
    if (R) run(); else dirty = true;
  }

  function setSteps(steps, go) {
    state.steps = clamp(steps, 1, M.maxSteps);
    const days = state.steps * TAU, inp = $("#stepsIn");
    inp.value = state.steps;
    inp.style.setProperty("--fill", `${(100 * (state.steps - 1)) / (M.maxSteps - 1)}%`);
    $("#daysOut").textContent = days;
    $("#daysHuman").textContent = humanDays(days);
    $$("#timeChips .chip").forEach((b) => b.classList.toggle("on", (!!R || go) && +b.dataset.steps === state.steps));
    if (go) run();
  }

  function setView(v) {
    state.view = v;
    $$(".mode").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === v));
    moveGlider();
    const fc = v === "forecast";
    $("#explorer").hidden = fc;
    $("#forecastPanel").hidden = !fc;
    $("#mapVeil").hidden = !fc || !!D.forecast;
    $("#timeline").style.visibility = fc ? "hidden" : "";
    $("#loopBtn").style.visibility = fc ? "hidden" : "";
    $("#legend").style.visibility = fc ? "hidden" : "";
    $("#mapHint").hidden = fc;
    $(".map-toggles").hidden = fc;
    if (fc) tip.classList.remove("show");
    state.highlight = null;
    if (fc) renderForecast();
    syncResults(); writeHash(); dirty = true;
    if (fc && D.forecast) state.highlight = "benguela"; // the area the Seasons chart is about
  }
  let gliderPlaced = false;
  function moveGlider() {
    const on = $(`.mode[data-view="${state.view}"]`), g = $(".mode-glider");
    if (!gliderPlaced) g.style.transition = "none"; // on load it starts in place, rather than sliding in from the corner
    g.style.left = `${on.offsetLeft}px`; g.style.width = `${on.offsetWidth}px`;
    if (!gliderPlaced) { void g.offsetWidth; g.style.transition = ""; gliderPlaced = true; }
  }

  // ---------- forecast view ----------
  function renderForecast() {
    const el = $("#forecastPanel"), F = D.forecast;
    if (!F) {
      el.innerHTML = `<span class="badge soon">In progress</span>
        <h2>Seasons</h2>
        <p>Does the drift change predictably with the seasons? This view will show the team's seasonal forecast: one year's seasonal pattern forecast from earlier years only, then checked against what actually happened.</p>
        <ol>
          <li><b>Pick a season</b> and see how the drift from the Benguela differs from the all-year picture.</li>
          <li><b>See the forecast</b> for the next season with its uncertainty, next to what was observed.</li>
          <li><b>Compare</b> with the drift explorer, which pools all seasons from ${D.meta.years[0]}–${D.meta.years[1]}.</li>
        </ol>
        <div class="team-note">For the team: save <code>data/processed/seasonal_forecast_app.json</code> in the format described in
        <code>src/app/export_app_data.py</code> and rerun <code>python3 -m src.app.export_app_data</code>. This panel then draws it.</div>`;
      return;
    }
    const W = 360, H = 190, l = 42, r = 34, t = 14, b = 30;
    const roundUp = (v) => { const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p; };
    const series = F.series || [];
    const xs = [...new Set(series.flatMap((s) => s.x))];
    const vals = series.flatMap((s) => [...s.y, ...(s.high || []), ...(s.low || [])]).filter(isFinite);
    const top = roundUp(Math.max(...vals) * 1.02), bot = Math.min(0, ...vals);
    const X1 = (v) => l + ((W - l - r) * xs.indexOf(v)) / Math.max(xs.length - 1, 1);
    const Y1 = (v) => t + (H - t - b) * (1 - (v - bot) / (top - bot));
    const colours = ["#38bdf8", "#fbbf24", "#2dd4bf", "#fb7185"];
    let g = "";
    for (let i = 0; i <= 4; i++) { const v = bot + ((top - bot) * i) / 4; g += `<line x1="${l}" x2="${W - r}" y1="${Y1(v)}" y2="${Y1(v)}"/><text x="${l - 6}" y="${Y1(v) + 3}" text-anchor="end">${+v.toPrecision(3)}</text>`; }
    const every = Math.ceil(xs.length / 6);
    const xl = xs.map((v, i) => (i % every ? "" : `<text x="${X1(v)}" y="${H - 10}" text-anchor="middle">${v}</text>`)).join("");
    const paths = series.map((s, i) => {
      const c = colours[i % colours.length];
      const ln = s.x.map((v, k) => `${k ? "L" : "M"}${X1(v)},${Y1(s.y[k])}`).join("");
      const band = s.low && s.high ? `<path class="band" fill="${c}" d="${s.x.map((v, k) => `${k ? "L" : "M"}${X1(v)},${Y1(s.high[k])}`).join("")}${s.x.map((v, k) => [X1(v), Y1(s.low[k])]).reverse().map(([a, bb]) => `L${a},${bb}`).join("")}Z"/>` : "";
      return `${band}<path class="line" stroke="${c}" d="${ln}" ${s.kind === "forecast" || s.dashed ? 'stroke-dasharray="6 4"' : ""}/>`;
    }).join("");
    el.innerHTML = `<span class="badge ready">Seasonal pattern</span><h2>${F.title || "Seasons"}</h2><p>${F.summary || ""}</p>
      <div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${F.quantity || "Forecast"}">${g}${xl}${paths}</svg></div>
      <div class="keys">${series.map((s, i) => `<span>${keyLine(s.kind === "forecast" || s.dashed, colours[i % colours.length], 2.4)}${s.label}</span>`).join("")}</div>
      ${F.quantity ? `<p class="fine">${F.quantity}${F.unit ? ` (${F.unit})` : ""}</p>` : ""}
      ${(F.notes || []).length ? `<ul class="fine">${F.notes.map((n) => `<li>${n}</li>`).join("")}</ul>` : ""}`;
  }

  // ---------- map interaction ----------
  const tip = $("#tooltip");
  const shoreIndex = new Map(D.shore.cells.map((c, k) => [c, k]));
  function cellAtEvent(e) {
    const rect = canvases.fx.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    return { cell: M.cellOf(latAt(y), lonAt(x)), x, y };
  }
  canvases.fx.addEventListener("pointermove", (e) => {
    if (state.view !== "explore") return;
    const { cell, x, y } = cellAtEvent(e);
    if (cell !== state.hover) { state.hover = cell; dirty = true; }
    wrap.classList.toggle("no-start", cell >= 0 && M.drifters[cell] === 0);
    if (cell < 0) { tip.classList.remove("show"); return; }
    const n = M.drifters[cell], flag = flagOf(n), step = R ? clamp(Math.round(state.t), 0, R.steps) : 0;
    const showing = R && step > 0 && !state.layers.coverage;
    const k = shoreIndex.get(cell);
    tip.innerHTML = `<b>${fmtLat(M.latC[cell])}, ${fmtLon(M.lonC[cell])}</b>
      ${showing ? `<div class="tt-row"><span>Chance it's ${R.ground ? "afloat " : ""}here on day ${step * TAU}</span><span>${fmtPct(100 * R.dists[step][cell])}</span></div>` : ""}
      ${showing && R.ground && k !== undefined ? `<div class="tt-row"><span>Chance it has run aground on this coast</span><span>${fmtPct(100 * R.dists[step][M.SHORE0 + k])}</span></div>` : ""}
      <div class="tt-row"><span>Drifters that started here</span><span>${n}${flag ? ` · ${flag.short}` : ""}</span></div>
      ${n === 0 ? `<div class="tt-no">No drifter started a move here (land or never sampled).</div>` : `<div class="tt-act">Click to start here</div>`}`;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = `${clamp(x + 16, 8, map.w - tw - 8)}px`;
    tip.style.top = `${clamp(y + 16 + th > map.h ? y - th - 12 : y + 16, 8, map.h - th - 8)}px`;
    tip.classList.add("show");
  });
  canvases.fx.addEventListener("pointerleave", () => { state.hover = -1; dirty = true; tip.classList.remove("show"); });
  function tryStart(cell) {
    if (cell < 0) return toast("That's outside the study area, where the model has no data.");
    if (M.drifters[cell] === 0) return toast("No drifter ever started a move in that square (land or never sampled). Try a nearby ocean square.");
    setStart(cellStart(cell));
  }
  canvases.fx.addEventListener("click", (e) => { if (state.view === "explore") tryStart(cellAtEvent(e).cell); });

  // ---------- typed positions ----------
  function parseCoord(text, axis) {
    const m = String(text).trim().toUpperCase().match(/^(-?\d+(?:\.\d+)?)\s*°?\s*([NSEW])?$/);
    if (!m) return NaN;
    let v = parseFloat(m[1]);
    if (m[2]) {
      if ((axis === "lat") !== "NS".includes(m[2])) return NaN;
      if ("SW".includes(m[2])) v = -Math.abs(v);
    }
    return v;
  }
  $("#coordForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const li = $("#latIn"), lo = $("#lonIn");
    const lat = parseCoord(li.value, "lat"), lon = parseCoord(lo.value, "lon");
    li.classList.toggle("bad", !isFinite(lat)); lo.classList.toggle("bad", !isFinite(lon));
    if (!isFinite(lat) || !isFinite(lon)) return toast("Type positions like 28S and 14E, or -28 and 14.");
    const cell = M.cellOf(lat, lon);
    if (cell < 0) return toast(`That position is outside the study area (${boxText(M.region)}).`);
    tryStart(cell);
  });

  let toastTimer = 0;
  function toast(msg) {
    const t = $("#toast");
    t.innerHTML = `<span>${msg}</span>`;
    t.hidden = false; t.style.animation = "none"; void t.offsetWidth; t.style.animation = "";
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), 4200);
  }

  // ---------- shareable links: #start=cell:103&days=90&ground=1 ----------
  function writeHash() {
    const h = state.view === "forecast" ? "view=forecast"
      : `start=${startKey(state.start)}&days=${state.steps * TAU}${state.layers.ground ? "&ground=1" : ""}`;
    try { history.replaceState(null, "", `#${h}`); } catch (e) { /* some browsers block this on file:// pages */ }
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    const out = { view: p.get("view") === "forecast" ? "forecast" : "explore", ground: p.get("ground") === "1" };
    const days = +p.get("days");
    if (days > 0) out.steps = clamp(Math.round(days / TAU), 1, M.maxSteps);
    const s = p.get("start") || "";
    if (s === "coast") out.start = coastStart();
    else if (/^cell:\d+$/.test(s) && M.drifters[+s.slice(5)] > 0) out.start = cellStart(+s.slice(5));
    return out;
  }

  // ---------- "How to read this" ----------
  function renderInfo() {
    const m = D.meta, Q = m.checks || {};
    const cal = Q.line_90 || ["about 90", "about 90", "about 85"], ex = Q.left_box || ["somewhat", "somewhat"], yr = Q.year_width || "1.5", c1 = Q.cell_1y || ["1", "1"], cov = Q.coverage;
    $("#infoBody").innerHTML = `
      <h3>Using it</h3>
      <ul>
        <li>Pick a start anywhere in the study area (${boxText(M.region)}), or release along the whole Benguela coast at once, then pick how long.
          The drift plays once and the results appear beside the map when it ends. Turn on <b>Loop</b> to keep replaying it, or drag the timeline to any day.</li>
      </ul>
      <h3>The map</h3>
      <ul>
        <li>Colour shows the chance the material is in each 2° square (about 200 km across) on the day shown. The bands are drawn as smooth contours
          through the squares' centres; <b>Grid cells</b> shows the squares the model really uses.</li>
        <li>The solid line encloses the smallest area the material has a 50% chance of being in, and the dashed line a 90% chance.
          Both count only material still in the water in the study area.</li>
        <li>The white diamond is the centre of the material, and the dotted line is the path it has taken.</li>
        <li><b>Groundings</b> adds the chance of running aground: the coastline turns pink where material has come ashore, brighter where more of it has.</li>
        <li><b>Data coverage</b> colours each square by its data flag (below).</li>
      </ul>
      <h3>Groundings</h3>
      <ul>
        <li>${m.n_grounded} drifters ran aground on a coast of the study area. A drifter that runs aground stops reporting, so the main model
          (and the report) leaves those last moves out, and material only drifts or leaves the box.</li>
        <li>With <b>Groundings</b> on, each square also has a chance of sending material ashore within 15 days, measured from those drifters,
          and the rest drifts as before. The numbers in the water are then a little lower.</li>
        <li>The chances come straight from the drifter counts, so coasts with few drifters have rough estimates; the 95% ranges show how rough.
          ${m.n_grounded_dropped} groundings away from the mapped coast (small islands, or likely record errors) are left out.</li>
      </ul>
      <h3>Data flags</h3>
      <ul>
        <li>Each start is flagged by how many drifters started a 15-day move from it. <b>Well sampled</b> means 100 or more, the level we treat as adequate.</li>
        <li><b>Limited data</b> (10–99) means the square's answer is partly filled in from the surrounding 4° area.</li>
        <li><b>Very thin data</b> (under 10) means the answer is a rough guide, and its 95% range will look narrower than it really is.</li>
      </ul>
      <h3>Where the numbers come from</h3>
      <ul>
        <li>${m.n_drifters.toLocaleString("en")} surface drifters of the Global Drifter Program (${m.years[0]}–${m.years[1]}), turned into ${m.n_moves.toLocaleString("en")}
          15-day moves between 2° squares, with thin squares borrowing from their 4° area (alpha ${m.alpha.toFixed(1)}). Longer times chain 15-day steps.</li>
        <li>The 95% ranges are a drifter bootstrap: the drifters are resampled with replacement ${m.n_boot.toLocaleString("en")} times, the model is refitted
          to each resample and run from the same start, and the range covers the middle 95% of the answers. The numbers shown are from the model fitted to all drifters.</li>
        <li>All seasons, and drifters with and without their drogue, are pooled. This is the typical near-surface drift of water and floating material, not a forecast for a particular date.
          Material more exposed to wind (e.g. oil or debris) tends to travel further west and leave more often.</li>
        <li>The ranges show the effect of which drifters we happened to have. They leave out how much the ocean changes from year to year, which would widen some of them: resampling whole years instead makes the range for the centre's longitude about ${yr} times as wide.</li>
      </ul>
      <h3>How far to trust it</h3>
      <ul>
        <li>Checked on drifters the model never saw. Chances of being in a square hold up out to a year: squares given about a ${c1[0]}% chance after a year really held drifters ${c1[1]}% of the time.</li>
        <li>The 90% line really held ${cal[0]}% of those drifters after 15 days, ${cal[1]}% after 3 months and ${cal[2]}% after a year, so treat it as a little optimistic at long times.</li>
        <li>The chance of having left the study area runs high at long times: ${ex[0]} after 3 months and ${ex[1]} after a year, across all starts.</li>
        ${cov ? `<li>In a simulation where the truth is known, the 95% ranges held it about ${cov[0]}% of the time from well-sampled squares, ${cov[1]}% from limited ones and ${cov[2]}% from very thin ones.
          For the whole-coast start it is only about ${cov[3]}%, because that start rests on thin squares, so its ranges are too narrow.</li>` : ""}
      </ul>`;
    $("#buildInfo").textContent = `Product: ${m.product} · data built ${m.built}`;
  }
  $("#infoBtn").addEventListener("click", () => $("#infoDialog").showModal());
  $("#infoClose").addEventListener("click", () => $("#infoDialog").close());
  $("#infoDialog").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); }); // click outside

  // ---------- wiring ----------
  TIME_CHIPS.forEach(([s, label]) => {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = label; b.dataset.steps = s; b.type = "button";
    b.onclick = () => setSteps(s, true);
    $("#timeChips").appendChild(b);
  });
  $("#stepsIn").max = M.maxSteps;
  $("#stepsIn").addEventListener("input", (e) => setSteps(+e.target.value, false));
  $("#stepsIn").addEventListener("change", (e) => setSteps(+e.target.value, true));
  $$(".mode").forEach((b) => b.addEventListener("click", () => b.dataset.view !== state.view && setView(b.dataset.view)));
  function setLayer(k, on) {
    state.layers[k] = on;
    const b = $(`.toggle[data-layer="${k}"]`);
    b.classList.toggle("on", on); b.setAttribute("aria-pressed", on);
    if (k === "ground") rerunInPlace();
    renderLegend(); dirty = true;
  }
  $$(".toggle").forEach((b) => b.addEventListener("click", () => setLayer(b.dataset.layer, !state.layers[b.dataset.layer])));
  $("#playBtn").addEventListener("click", () => {
    if (!R) return;
    if (state.playing) { state.playing = false; clearTimeout(loopTimer); }
    else play(state.t >= R.steps);
  });
  $("#loopBtn").addEventListener("click", (e) => {
    state.loop = !state.loop;
    e.currentTarget.setAttribute("aria-pressed", state.loop);
    clearTimeout(loopTimer);
    if (state.loop && R && !state.playing && state.t >= R.steps) play(true);
  });
  $("#scrub").addEventListener("input", (e) => {
    clearTimeout(loopTimer);
    state.t = +e.target.value; state.playing = false; dirty = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, button, select, textarea, dialog") || state.view !== "explore") return;
    if (e.code === "Space" && R) { e.preventDefault(); $("#playBtn").click(); }
  });
  new ResizeObserver(moveGlider).observe($(".modes"));
  new ResizeObserver(resize).observe(wrap);
  window.addEventListener("resize", fitSide);

  let lastFrame = performance.now(), lastT = -1;
  function frame(now) {
    const dt = clamp(now - lastFrame, 0, 100); // a frame's timestamp can be a little before performance.now() at load
    lastFrame = now;
    if (R && state.playing) {
      state.t = clamp(state.t + dt / stepMs(R.steps), 0, R.steps);
      if (state.t >= R.steps) onPlayEnd();
      dirty = true;
    }
    if (dirty) {
      if (state.t !== lastT || !state.playing) updateTimeUI();
      lastT = state.t;
      syncResults();
      if (map.w) { drawHeat(); drawFx(); }
      dirty = false;
    }
    requestAnimationFrame(frame);
  }

  // ---------- start ----------
  renderInfo();
  if (D.forecast) $("#forecastTabSub").textContent = "Through the year";
  const init = readHash();
  if (init.start) state.start = init.start;
  if (init.ground) setLayer("ground", true);
  setSteps(init.steps || 6, false);
  $("#timeStep").classList.add("await");
  $("#mapHint").textContent = "Choose a start on the map, then pick how long";
  resize(); fitSide();
  renderStart();
  if (init.start && init.steps) run();
  setView(init.view);
  renderLegend(); updateTimeUI();
  requestAnimationFrame(frame);
  window.__drift = { MODELS, model: () => M, state, run: () => R }; // for the console
})();
