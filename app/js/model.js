// model.js - the maths behind the Drift Explorer, on window.DRIFT_DATA (made by
// src/app/export_app_data.py). No page code here, so tests/app_check.html can test it against Python.
//
// States are the 2 deg cells of the box (row-major from the south-west corner, as grid.cell_index),
// then "left the box" (absorbing). A Model is one of the app's matrices:
// - drifters "all" is the product, "undrogued" the same fit on moves that start without a drogue.
// - variant "shore" adds one absorbing "ran aground" state per stretch of coast: each cell's row is
//   scaled by its chance of not running aground within 15 days, and that chance is shared out over the
//   coasts (see the exporter's docstring).
// - a release season: a season's row is the all-year row pulled towards that season's moves,
//   (n^s + beta P) / (n^s_i + beta), and a chain changes season every per_season steps
//   (src/forecasting/seasonal_matrices.py). Built when first used. The grounding chance stays the all-year one.
// One step is 15 days: dist_next = dist @ P, exactly Product.forecast in src/linalg/product.py. P is shipped
// as float32, so each row is renormalised to sum to 1 after decoding (a change of under 1e-6 per entry).
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

  // Intervals are 16 bits on a log scale: 0 is "too small to matter", 65535 is "no value" (the exporter's log16).
  function decodeLog16(b64, range) {
    const bin = atob(b64), n = bin.length / 2, out = new Float32Array(n), span = range[1] - range[0];
    for (let i = 0; i < n; i++) {
      const code = bin.charCodeAt(2 * i) + 256 * bin.charCodeAt(2 * i + 1);
      out[i] = code === 0 ? 0 : code === 65535 ? NaN : 10 ** (range[0] + (code / 65534) * span);
    }
    return out;
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
    constructor(d, variant = "water", drifters = "all") {
      this.data = d;
      this.variant = variant;
      this.drifterSet = drifters;
      const set = d.sets[drifters];
      this.set = set;
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
      this.perSeason = d.seasons.per_season;

      const m = this.nCells + 1, n = this.nStates;
      const P32 = decodeFloat32(set.P);
      if (P32.length !== m * m) throw new Error("P has the wrong size for this grid.");
      this.afloat = new Float64Array(m).fill(1);
      if (variant === "shore") for (const [i, , v] of set.ground_rows) this.afloat[i] -= v;
      this.water = new Float64Array(m * m); // the all-year rows before any chance of running aground
      this.P = new Float64Array(n * n);
      for (let i = 0; i < m; i++) {
        let s = 0;
        for (let j = 0; j < m; j++) s += P32[i * m + j];
        for (let j = 0; j < m; j++) {
          this.water[i * m + j] = P32[i * m + j] / s;
          this.P[i * n + j] = this.afloat[i] * this.water[i * m + j];
        }
      }
      if (variant === "shore") {
        for (const [i, k, v] of set.ground_rows) this.P[i * n + this.SHORE0 + k] = v;
        for (let k = 0; k < this.nShore; k++) this.P[(this.SHORE0 + k) * n + this.SHORE0 + k] = 1;
      }
      this.seasonP = [null, null, null, null];

      this.latC = new Float64Array(this.nCells);
      this.lonC = new Float64Array(this.nCells);
      for (let c = 0; c < this.nCells; c++) {
        const r = Math.floor(c / this.nLon), k = c % this.nLon;
        this.latC[c] = this.region.lat_min + (r + 0.5) * this.size;
        this.lonC[c] = this.region.lon_min + (k + 0.5) * this.size;
      }
      this.drifters = Int32Array.from(set.start_drifters);
      this.allLand = new Uint8Array(this.nCells);
      (d.all_land || []).forEach((c) => (this.allLand[c] = 1));
      this.inBenguela = new Uint8Array(this.nCells);
      d.benguela_cells.forEach((c) => (this.inBenguela[c] = 1));
      this.isCoastal = new Uint8Array(this.nCells);
      d.coastal_cells.forEach((c) => (this.isCoastal[c] = 1));
      this.releaseCells = set.release_cells.slice();

      this.intervalData = new Map(); // decoded when first used: "all" or a season index -> low, high
    }

    // The matrix of one season (0-3, in the order of d.seasons), built when first used.
    seasonMatrix(s) {
      if (this.seasonP[s]) return this.seasonP[s];
      const m = this.nCells + 1, n = this.nStates, { beta, counts } = this.set.seasonal;
      const pairs = counts[this.data.seasons.codes[s]], totals = new Float64Array(m);
      for (let p = 0; p < pairs.length; p += 2) totals[Math.floor(pairs[p] / m)] += pairs[p + 1];
      const out = Float64Array.from(this.P); // running aground, and its absorbing states, stay as they are
      for (let i = 0; i < m; i++) {
        const w = (this.afloat[i] * beta) / (totals[i] + beta);
        for (let j = 0; j < m; j++) out[i * n + j] = w * this.water[i * m + j];
      }
      for (let p = 0; p < pairs.length; p += 2) {
        const i = Math.floor(pairs[p] / m), j = pairs[p] % m;
        out[i * n + j] += (this.afloat[i] * pairs[p + 1]) / (totals[i] + beta);
      }
      return (this.seasonP[s] = out);
    }

    // The season of step `step` (0 = the first) of a release at the first day of `season`.
    seasonAt(season, step) {
      return (season + Math.floor(step / this.perSeason)) % 4;
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

    // Distributions after 0, 1, ..., nSteps steps, all year (season null) or from the first day of a season.
    propagate(x0, nSteps, season = null) {
      const n = this.nStates, out = [Float64Array.from(x0)];
      for (let s = 0; s < nSteps; s++) {
        const P = season === null ? this.P : this.seasonMatrix(this.seasonAt(season, s));
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

    // % that has entered the Benguela coastal cells at any time up to each step: material that arrives is kept
    // there (first passage, as src/linalg/connectivity.py). Drifters that run aground stop reporting, so without
    // the shore model this is a lower bound.
    reachedCoast(x0, nSteps, season = null) {
      const n = this.nStates;
      let x = Float64Array.from(x0);
      const share = (v) => { let t = 0; for (let c = 0; c < this.nCells; c++) if (this.isCoastal[c]) t += v[c]; return 100 * t; };
      const out = [share(x)];
      for (let s = 0; s < nSteps; s++) {
        const P = season === null ? this.P : this.seasonMatrix(this.seasonAt(season, s));
        const y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const xi = x[i];
          if (xi === 0) continue;
          if (i < this.nCells && this.isCoastal[i]) { y[i] += xi; continue; }
          const off = i * n;
          for (let j = 0; j < n; j++) y[j] += xi * P[off + j];
        }
        out.push(share(y));
        x = y;
      }
      return out;
    }

    // % of a distribution in each subregion (src/linalg/connectivity.py), in the order of d.subregions.names.
    subregionShares(dist) {
      const { names, label } = this.data.subregions, out = new Array(names.length).fill(0);
      for (let c = 0; c < this.nCells; c++) out[label[c]] += 100 * dist[c];
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

    // The decoded intervals of a run: all year (season null) or from the first day of a season.
    intervalsOf(season) {
      const key = season === null ? "all" : season;
      if (!this.intervalData.has(key)) {
        const seasonal = season !== null, iv = (seasonal ? this.set.seasonal.intervals : this.set.intervals)[this.variant];
        const range = this.data.log16;
        this.intervalData.set(key, {
          quantities: iv.quantities, steps: iv.steps, nRows: iv.rows.length, rows: new Map(iv.rows.map((c, k) => [c, k])),
          stepAt: new Map(iv.steps.map((n, k) => [n, k])),
          low: decodeLog16(seasonal ? iv.low[season] : iv.low, range), high: decodeLog16(seasonal ? iv.high[season] : iv.high, range),
        });
      }
      return this.intervalData.get(key);
    }

    // 95% bootstrap interval [low, high] of a quantity after `step` steps (null at step 0 or if not exported). A season's
    // intervals are stored at some steps only, so between those it is read off a straight line on a log scale.
    interval(start, quantity, step, season = null) {
      const iv = this.intervalsOf(season);
      const row = iv.rows.get(start.kind === "coast" ? -1 : start.cell);
      const q = iv.quantities.indexOf(quantity);
      if (row === undefined || q < 0 || step < 1 || step > iv.steps[iv.steps.length - 1]) return null;
      const at = (k) => {
        const i = (q * iv.steps.length + k) * iv.nRows + row;
        return [iv.low[i], iv.high[i]];
      };
      const exact = iv.stepAt.get(step);
      if (exact !== undefined) return at(exact);
      let k = 0;
      while (iv.steps[k + 1] < step) k++;
      const f = (step - iv.steps[k]) / (iv.steps[k + 1] - iv.steps[k]), a = at(k), b = at(k + 1);
      return a.map((v, i) => (v > 0 && b[i] > 0 ? 10 ** (Math.log10(v) * (1 - f) + Math.log10(b[i]) * f) : v * (1 - f) + b[i] * f));
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

  global.DriftModel = { Model, decodeFloat32, decodeLog16, greatCircleKm, bearingDeg, compass };
})(window);
