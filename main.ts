// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate at the
// case's target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) in
// percent. It improves on the house baseline (persistence + random-walk
// spread) in two ways:
//
// 1. Median: a per-trophy trend fit. Following the house curve baseline, it
//    fits a least-squares line to logit(rate) against ln(1 + hours since
//    2026-11-19T00:00:00Z) over the case's last 24 hours of reads, and reads
//    the line at the target. The extrapolated move is capped at 2.5 logit
//    units, and the forecast median blends the trend with persistence
//    (the last read's rate) with weight 0.85 * R^2 of the fit, so a flat or
//    noisy history stays close to persistence while a real trend is
//    followed. With fewer than 3 usable reads, or a history with no
//    variation at all, it falls back to persistence exactly.
// 2. Spread: an honest one, so the 90% band covers about 90% of outcomes.
//    sigma is the root-sum-square of three logit-scale terms:
//      - random walk: 0.5 * sqrt(horizon in days) (the house baseline's);
//      - fit uncertainty: the fit's residual std times the prediction
//        leverage at the target (1/n + (x_target - xbar)^2 / Sxx),
//        capped at 1.5; 0.25 when no fit exists, because an unseen drift
//        is still possible;
//      - regime allowance: 0.25 + 0.15 * sqrt(horizon in days), for
//        Exophase refresh timing and player-base composition shifts the
//        history cannot show. The practice cases are the warning here:
//        every read before their targets is the same flat launch-night
//        page, and the answer is the first refreshed page, so a spread
//        built only from observed movement is far too narrow.
//
// Everything the program knows is in the case file: it runs offline, uses
// no randomness and no clock, so two runs on the same input write the same
// bytes. It reads every <id>.in in the input directory and writes
// <id>.out to the output directory: {"trophies":[{"id":n,
// "q":[q05,q25,q50,q75,q95]}, ...]}, one row per trophy, each q five
// finite numbers from 0 to 100, non-decreasing, at most 65,536 bytes.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and
// MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Time origin of the log-time trend fit (release day, house curve baseline). */
const T0_MS = Date.parse('2026-11-19T00:00:00Z');
/** How far back the trend fit looks, in milliseconds. */
const FIT_WINDOW_MS = 24 * 60 * 60 * 1000;
/** Random-walk spread, in logit units per square-root day. */
const SIGMA_RW = 0.5;
/** Regime/refresh allowance: base logit units, plus this per square-root day. */
const REGIME_BASE = 0.25;
const REGIME_PER_SQRT_DAY = 0.15;
/** Spread allowed to a trophy whose history admits no trend fit. */
const NO_FIT_SIGMA = 0.25;
/** Cap on the fit-uncertainty term, in logit units. */
const FIT_SIGMA_CAP = 1.5;
/** Cap on how far the trend may move the median from the last rate, logit units. */
const TREND_MOVE_CAP = 2.5;
/** Largest weight the trend fit can take in the median blend. */
const MAX_TREND_WEIGHT = 0.85;

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  rates: Array<number | null>;
}
interface Case {
  at: string;
  levels: number[];
  horizon_minutes: number;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/**
 * Inverse standard normal CDF (Acklam's rational approximation, ~1e-9
 * accurate). Used only when a case asks for levels other than the
 * contest's five, whose z values come from the Z table above.
 */
const normInv = (p: number): number => {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
};

/** ln(1 + hours from the release-day origin to a read), clamped at 0. */
const logTime = (atMs: number): number => Math.log(1 + Math.max(0, (atMs - T0_MS) / 3_600_000));

/** Least-squares line through (x, y) points, plus fit diagnostics. */
const fitLine = (xs: number[], ys: number[]): {
  slope: number;
  intercept: number;
  r2: number;
  residStd: number;
  xbar: number;
  sxx: number;
  n: number;
} | null => {
  const n = xs.length;
  if (n < 3) return null;
  let xbar = 0;
  let ybar = 0;
  for (let i = 0; i < n; i++) {
    xbar += xs[i];
    ybar += ys[i];
  }
  xbar /= n;
  ybar /= n;
  let sxx = 0;
  let sxy = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xbar;
    const dy = ys[i] - ybar;
    sxx += dx * dx;
    sxy += dx * dy;
    sst += dy * dy;
  }
  // No variation in x or in y: no trend to fit.
  if (sxx <= 0 || sst <= 0) return null;
  const slope = sxy / sxx;
  const intercept = ybar - slope * xbar;
  let ssr = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    ssr += e * e;
  }
  return {
    slope,
    intercept,
    r2: clamp(1 - ssr / sst, 0, 1),
    residStd: Math.sqrt(ssr / (n - 2)),
    xbar,
    sxx,
    n,
  };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const levels = c.levels.length > 0 ? c.levels : LEVELS;
  const useStandardZ = levels.length === LEVELS.length && levels.every((t, j) => t === LEVELS[j]);
  const z = useStandardZ ? Z : levels.map((t) => normInv(clamp(t, 0.0001, 0.9999)));

  const checkpointMs = Date.parse(c.at);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const sqrtDays = Math.sqrt(horizonDays);
  const targetMs = Number.isFinite(checkpointMs)
    ? checkpointMs + Math.max(0, c.horizon_minutes) * 60_000
    : T0_MS;
  const xTarget = logTime(targetMs);
  const windowStart = Number.isFinite(checkpointMs) ? checkpointMs - FIT_WINDOW_MS : -Infinity;

  // Per-trophy series, newest first, for the latest non-null rate.
  const trophies = c.trophies.map((t, i) => {
    let lastRate: number | null = null;
    for (let h = c.history.length - 1; h >= 0; h--) {
      const r = c.history[h].rates[i];
      if (typeof r === 'number' && Number.isFinite(r)) {
        lastRate = r;
        break;
      }
    }
    // No read ever listed this trophy: centre low with a wide spread.
    if (lastRate === null) {
      const l = logit(1);
      return {
        id: t.id,
        q: z.map((zz) => round4(clamp(invLogit(l + zz * 2), 0, 100))),
      };
    }
    const lastLogit = logit(lastRate);

    // Trend fit on the last 24 hours of reads.
    const xs: number[] = [];
    const ys: number[] = [];
    for (const read of c.history) {
      const atMs = Date.parse(read.at);
      if (Number.isFinite(atMs) && atMs < windowStart) continue;
      const r = read.rates[i];
      if (typeof r !== 'number' || !Number.isFinite(r)) continue;
      xs.push(logTime(atMs));
      ys.push(logit(r));
    }
    const fit = fitLine(xs, ys);

    let medianLogit = lastLogit;
    let fitSigma = NO_FIT_SIGMA;
    if (fit !== null) {
      const trendLogit = fit.intercept + fit.slope * xTarget;
      const move = clamp(trendLogit - lastLogit, -TREND_MOVE_CAP, TREND_MOVE_CAP);
      const w = MAX_TREND_WEIGHT * fit.r2;
      medianLogit = (1 - w) * lastLogit + w * (lastLogit + move);
      const leverage = 1 / fit.n + ((xTarget - fit.xbar) ** 2) / fit.sxx;
      fitSigma = Math.min(FIT_SIGMA_CAP, fit.residStd * Math.sqrt(Math.max(0, leverage)));
    }

    const sigma = Math.sqrt(
      (SIGMA_RW * sqrtDays) ** 2 +
        fitSigma ** 2 +
        (REGIME_BASE + REGIME_PER_SQRT_DAY * sqrtDays) ** 2,
    );
    const q = z.map((zz) => round4(clamp(invLogit(medianLogit + zz * sigma), 0, 100)));
    // Rounding must not break monotonicity.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
