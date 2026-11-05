// model.js - the maths behind the Drift Explorer, on window.DRIFT_DATA (made by
// src/app/export_app_data.py). No page code here, so tests/app_check.html can test it against Python.
//
// States are the 2 deg cells of the box (row-major from the south-west corner, as grid.cell_index),
// then "left the box" (absorbing). The "shore" variant adds one absorbing "ran aground" state per
// stretch of coast: each cell's product row is scaled by its chance of not running aground within
// 15 days, and that chance is shared out over the coasts (see the exporter's docstring). One step is
// 15 days: dist_next = dist @ P, exactly Product.forecast in src/linalg/product.py for the "water"
// variant. P is shipped as float32, so each product row is renormalised to sum to 1 after decoding
// (a change of under 1e-6 per entry).
(function (global) {
  "use strict";

  const EARTH_RADIUS_KM = 6371;
  const COMPASS = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast",
    "southeast", "south-southeast", "south", "south-southwest", "southwest", "west-southwest", "west",
    "west-northwest", "northwest", "north-northwest"];

  function decodeFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  const rad = (d) => (d * Math.PI) / 180;

  function greatCircleKm(lat1, lon1, lat2, lon2) {
    const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 +
      Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
    return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(Math.min(1, Math.max(0, a))));
  }

  function bearingDeg(lat1, lon1, lat2, lon2) {
    const y = Math.sin(rad(lon2 - lon1)) * Math.cos(rad(lat2));
    const x = Math.cos(rad(lat1)) * Math.sin(rad(lat2)) - Math.sin(rad(lat1)) * Math.cos(rad(lat2)) * Math.cos(rad(lon2 - lon1));
    return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
  }

  const compass = (deg) => COMPASS[Math.round(deg / 22.5) % 16];

  class Model {
    constructor(d, variant = "water") {
      this.data = d;
      this.variant = variant;
      this.nLat = d.n_lat;
      this.nLon = d.n_lon;
      this.nCells = d.n_lat * d.n_lon;
      this.LEFT = this.nCells;
      this.size = d.cell_size;
      this.tau = d.tau_days;
      this.maxSteps = d.max_steps;
      this.region = d.region;
      this.benguela = d.benguela;
      this.shoreCells = variant === "shore" ? d.shore.cells.slice() : [];
      this.nShore = this.shoreCells.length;
      this.SHORE0 = this.nCells + 1; // first "ran aground" state
      this.nStates = this.nCells + 1 + this.nShore;

      const m = this.nCells + 1, n = this.nStates;
      const P32 = decodeFloat32(d.P);
      if (P32.length !== m * m) throw new Error("P has the wrong size for this grid.");
      const afloat = new Float64Array(m).fill(1);
      if (variant === "shore") for (const [i, , v] of d.shore.rows) afloat[i] -= v;
      this.P = new Float64Array(n * n);
      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let j = 0; j < m; j++) s += P32[i * m + j];
        for (let j = 0; j < m; j++) this.P[i * n + j] = (afloat[i] * P32[i * m + j]) / s;
      }
      if (variant === "shore") {
        for (const [i, k, v] of d.shore.rows) this.P[i * n + this.SHORE0 + k] = v;
        for (let k = 0; k < this.nShore; k++) this.P[(this.SHORE0 + k) * n + this.SHORE0 + k] = 1;
      }

      this.latC = new Float64Array(this.nCells);
      this.lonC = new Float64Array(this.nCells);
      for (let c = 0; c < this.nCells; c++) {
        const r = Math.floor(c / this.nLon), k = c % this.nLon;
        this.latC[c] = this.region.lat_min + (r + 0.5) * this.size;
        this.lonC[c] = this.region.lon_min + (k + 0.5) * this.size;
      }
      this.drifters = Int32Array.from(d.start_drifters);
      this.allLand = new Uint8Array(this.nCells);
      (d.all_land || []).forEach((c) => (this.allLand[c] = 1));
      this.inBenguela = new Uint8Array(this.nCells);
      d.benguela_cells.forEach((c) => (this.inBenguela[c] = 1));
      this.isCoastal = new Uint8Array(this.nCells);
      d.coastal_cells.forEach((c) => (this.isCoastal[c] = 1));
      this.releaseCells = d.release_cells.slice();

      const iv = d.intervals[variant];
      this.quantities = iv.quantities;
      this.ivSteps = iv.steps;
      this.ivRows = iv.rows.length;
      this.ivRow = new Map(iv.rows.map((c, k) => [c, k]));
      this.ivLow = decodeFloat32(iv.low);
      this.ivHigh = decodeFloat32(iv.high);
    }

    // Flat cell id of a position, or -1 outside the box.
    cellOf(lat, lon) {
      const g = this.region;
      if (!(lat >= g.lat_min && lat <= g.lat_max && lon >= g.lon_min && lon <= g.lon_max)) return -1;
      const r = Math.min(this.nLat - 1, Math.floor((lat - g.lat_min) / this.size));
      const k = Math.min(this.nLon - 1, Math.floor((lon - g.lon_min) / this.size));
      return r * this.nLon + k;
    }

    cellBounds(c) {
      const r = Math.floor(c / this.nLon), k = c % this.nLon;
      const lat0 = this.region.lat_min + r * this.size, lon0 = this.region.lon_min + k * this.size;
      return { lat0, lat1: lat0 + this.size, lon0, lon1: lon0 + this.size };
    }

    cellAreaKm2(c) {
      const b = this.cellBounds(c);
      return EARTH_RADIUS_KM ** 2 * rad(this.size) * Math.abs(Math.sin(rad(b.lat1)) - Math.sin(rad(b.lat0)));
    }

    // A start is {kind: "cell", cell} or {kind: "coast"} (spread evenly over the coastal cells with
    // drifters, as src/linalg/coastal_release.py).
    startVector(start) {
      const x = new Float64Array(this.nStates);
      if (start.kind === "coast") this.releaseCells.forEach((c) => (x[c] = 1 / this.releaseCells.length));
      else x[start.cell] = 1;
      return x;
    }

    startPoint(start) {
      if (start.kind !== "coast") return { lat: this.latC[start.cell], lon: this.lonC[start.cell] };
      const cells = this.releaseCells;
      return {
        lat: cells.reduce((s, c) => s + this.latC[c], 0) / cells.length,
        lon: cells.reduce((s, c) => s + this.lonC[c], 0) / cells.length,
      };
    }

    // Distributions after 0, 1, ..., nSteps steps.
    propagate(x0, nSteps) {
      const n = this.nStates, P = this.P, out = [Float64Array.from(x0)];
      for (let s = 0; s < nSteps; s++) {
        const x = out[s], y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const xi = x[i];
          if (xi === 0) continue;
          const off = i * n;
          for (let j = 0; j < n; j++) y[j] += xi * P[off + j];
        }
        out.push(y);
      }
      return out;
    }

    // The exporter's QUANTITIES for one distribution (shares in %, centre over material still in the
    // water inside the box).
    stats(dist, start) {
      let inside = 0, lat = 0, lon = 0, beng = 0, coast = 0, aground = 0;
      for (let c = 0; c < this.nCells; c++) {
        const p = dist[c];
        if (p === 0) continue;
        inside += p;
        lat += p * this.latC[c];
        lon += p * this.lonC[c];
        if (this.inBenguela[c]) beng += p;
        if (this.isCoastal[c]) coast += p;
      }
      for (let s = this.SHORE0; s < this.nStates; s++) aground += dist[s];
      const o = this.startPoint(start);
      const centreLat = lat / inside, centreLon = lon / inside;
      return {
        pct_left_box: 100 * dist[this.LEFT], pct_in_benguela: 100 * beng, pct_in_coastal_zone: 100 * coast,
        centre_lat: centreLat, centre_lon: centreLon, centre_km: greatCircleKm(o.lat, o.lon, centreLat, centreLon),
        pct_aground: 100 * aground, inside,
      };
    }

    // 95% bootstrap interval [low, high] of a quantity after `step` steps (null at step 0 or if not exported).
    interval(start, quantity, step) {
      const row = this.ivRow.get(start.kind === "coast" ? -1 : start.cell);
      const q = this.quantities.indexOf(quantity);
      if (row === undefined || q < 0 || step < 1 || step > this.ivSteps) return null;
      const i = (q * this.ivSteps + (step - 1)) * this.ivRows + row;
      return [this.ivLow[i], this.ivHigh[i]];
    }

    // Smallest set of cells holding each share of the material still in the water in the box (most
    // likely cells first). pMin is the smallest cell in the set, pNext the largest one left out.
    likelyAreas(dist, levels) {
      const order = [];
      let inside = 0;
      for (let c = 0; c < this.nCells; c++) if (dist[c] > 0) { order.push(c); inside += dist[c]; }
      order.sort((a, b) => dist[b] - dist[a]);
      return levels.map((level) => {
        const mask = new Uint8Array(this.nCells);
        let acc = 0, cells = 0, km2 = 0, i = 0;
        for (; i < order.length && acc < level * inside; i++) {
          const c = order[i];
          mask[c] = 1;
          acc += dist[c];
          cells++;
          km2 += this.cellAreaKm2(c);
        }
        return { level, mask, cells, km2, pMin: dist[order[i - 1]], pNext: i < order.length ? dist[order[i]] : 0 };
      });
    }
  }

  global.DriftModel = { Model, decodeFloat32, greatCircleKm, bearingDeg, compass };
})(window);
