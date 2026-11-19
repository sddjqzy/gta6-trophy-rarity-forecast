// GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01.
// Node 22, standard library only. No network, no randomness, no clock, so
// the same input always produces the same output bytes.
//
// For every trophy in a case, this program predicts the Exophase earn rate
// (percent of tracked PS5 players who have earned it) at the case's target
// instant, as five quantiles (5% / 25% / 50% / 75% / 95%). Two ideas:
//
// * Centre: a per-trophy log-time trend. Each recorded earn rate is turned
//   into its logit, an ordinary least-squares line is fitted to
//   logit(rate) against ln(1 + hours since 2026-11-19T00:00:00Z) using the
//   reads from the 24 hours before the checkpoint, and the line is read at
//   the target. The trend's pull away from the latest observed rate is
//   limited to 2.5 logit units, and the centre blends the trend with plain
//   persistence (the latest rate), weighted by 0.85 * R^2 — flat or noisy
//   histories stay near persistence while a genuine trend is followed.
//   Fewer than three usable reads, or a history with no variation at all,
//   falls back to persistence.
// * Width: an honest spread on the logit scale, combining three terms in
//   quadrature: a random-walk term (0.5 * sqrt(days of horizon), as in the
//   house baseline), the trend fit's own prediction uncertainty (residual
//   spread times the leverage of the target point, capped at 1.5; 0.25 when
//   there is no fit, since drift the history cannot see is still
//   possible), and a regime term (0.25 + 0.15 * sqrt(days)) for Exophase
//   refresh timing and shifts in who the tracked players are. The practice
//   cases are the cautionary tale here: every read before their targets
//   shows the same flat launch-night page and the answer is the first
//   refreshed page, so a spread built only from observed movement would be
//   far too narrow.
//
// The program reads every *.in file in the input directory (in name order)
// and writes one <id>.out per case: {"trophies":[{"id":n,"q":[q05,q25,q50,
// q75,q95]}, ...]}. Each quantile is a finite number in [0, 100],
// non-decreasing within a row, rounded to 4 decimals, and the file stays
// under the 65,536-byte cap.
//
// Run locally as:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
// which stand in for /input and /output outside the grader's container.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels every row reports, in order. */
const QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles matching QUANTILE_LEVELS. */
const NORMAL_Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Start of the log-time axis: release day, as in the house curve baseline. */
const EPOCH_MS = Date.parse('2026-11-19T00:00:00Z');
/** How far back the trend fit reaches, in milliseconds. */
const LOOKBACK_MS = 24 * 60 * 60 * 1000;
/** Random-walk spread per square-root day, in logit units. */
const WALK_SIGMA = 0.5;
/** Regime allowance: a base amount plus this per square-root day. */
const REGIME_BASE = 0.25;
const REGIME_PER_DAY = 0.15;
/** Spread for a trophy whose history supports no trend fit. */
const NO_FIT_SIGMA = 0.25;
/** Upper bound on the fit-uncertainty term, in logit units. */
const FIT_SIGMA_CAP = 1.5;
/** Largest logit move the trend may pull the centre from the latest rate. */
const TREND_CAP = 2.5;
/** Largest share of the centre the trend fit may claim. */
const MAX_TREND_SHARE = 0.85;
/** Spread used when a trophy never appears in any read. */
const UNSEEN_SIGMA = 2;

interface TrophyEntry {
  id: number;
}
interface Snapshot {
  at: string;
  rates: Array<number | null>;
}
interface CaseFile {
  at: string;
  levels: number[];
  horizon_minutes: number;
  trophies: TrophyEntry[];
  history: Snapshot[];
}

const clampTo = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampPercent = (x: number): number => clampTo(x, 0.01, 99.99);
const toLogit = (pct: number): number => {
  const p = clampPercent(pct) / 100;
  return Math.log(p / (1 - p));
};
const fromLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimal places. */
const round4dp = (x: number): number => Math.round(x * 10_000) / 10_000;

/**
 * Blend persistence with the trend: put weight w on the trend's (capped)
 * move away from the latest rate, and the rest on staying put.
 */
const blendCentre = (lastLogit: number, trendLogit: number, w: number): number => {
  const move = clampTo(trendLogit - lastLogit, -TREND_CAP, TREND_CAP);
  return (1 - w) * lastLogit + w * (lastLogit + move);
};

/** Evaluate a polynomial at x by Horner's rule. */
const horner = (coeffs: number[], x: number): number => {
  let acc = 0;
  for (const c of coeffs) acc = acc * x + c;
  return acc;
};

/**
 * Inverse standard normal CDF (Acklam's rational approximation, ~1e-9
 * accurate). Only needed when a case asks for levels other than the
 * contest's five, whose z values come from the NORMAL_Z table.
 */
const normInv = (p: number): number => {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285984469534e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const tail = 0.02425;
  if (p < tail) {
    const q = Math.sqrt(-2 * Math.log(p));
    return horner(c, q) / (horner(d, q) * q + 1);
  }
  if (p > 1 - tail) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(horner(c, q) / (horner(d, q) * q + 1));
  }
  const q = p - 0.5;
  const r = q * q;
  return (horner(a, r) * q) / (horner(b, r) * r + 1);
};

/** ln(1 + hours from the release-day origin to a read), floored at 0. */
const logHours = (atMs: number): number => Math.log(1 + Math.max(0, (atMs - EPOCH_MS) / 3_600_000));

interface TrendFit {
  slope: number;
  intercept: number;
  rSquared: number;
  residualStd: number;
  meanX: number;
  sumSqX: number;
  count: number;
}

/** Least-squares line through (x, y) points, plus fit diagnostics. */
const leastSquares = (xs: number[], ys: number[]): TrendFit | null => {
  const n = xs.length;
  if (n < 3) return null;
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += xs[i];
    meanY += ys[i];
  }
  meanX /= n;
  meanY /= n;
  let sumSqX = 0;
  let sumCross = 0;
  let sumSqY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    sumSqX += dx * dx;
    sumCross += dx * dy;
    sumSqY += dy * dy;
  }
  // No spread in x or in y: there is no trend to fit.
  if (sumSqX <= 0 || sumSqY <= 0) return null;
  const slope = sumCross / sumSqX;
  const intercept = meanY - slope * meanX;
  let ssr = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    ssr += e * e;
  }
  return {
    slope,
    intercept,
    rSquared: clampTo(1 - ssr / sumSqY, 0, 1),
    residualStd: Math.sqrt(ssr / (n - 2)),
    meanX,
    sumSqX,
    count: n,
  };
};

/** Values shared by every trophy forecast within one case. */
interface CaseContext {
  z: number[];
  xTarget: number;
  windowStart: number;
  sqrtDays: number;
}

/** The latest non-null rate for one trophy, scanning newest first. */
const latestRate = (history: Snapshot[], i: number): number | null => {
  for (let h = history.length - 1; h >= 0; h--) {
    const r = history[h].rates[i];
    if (typeof r === 'number' && Number.isFinite(r)) return r;
  }
  return null;
};

/** One trophy's five-quantile forecast row. */
const forecastTrophy = (
  trophy: TrophyEntry,
  i: number,
  c: CaseFile,
  ctx: CaseContext,
): { id: number; q: number[] } => {
  const last = latestRate(c.history, i);
  // No read ever listed this trophy: centre low with a wide spread.
  if (last === null) {
    const centre = toLogit(1);
    return {
      id: trophy.id,
      q: ctx.z.map((zz) => round4dp(clampTo(fromLogit(centre + zz * UNSEEN_SIGMA), 0, 100))),
    };
  }
  const lastLogit = toLogit(last);

  // Trend fit on the last 24 hours of reads.
  const xs: number[] = [];
  const ys: number[] = [];
  for (const read of c.history) {
    const atMs = Date.parse(read.at);
    if (Number.isFinite(atMs) && atMs < ctx.windowStart) continue;
    const r = read.rates[i];
    if (typeof r !== 'number' || !Number.isFinite(r)) continue;
    xs.push(logHours(atMs));
    ys.push(toLogit(r));
  }
  const fit = leastSquares(xs, ys);

  let centreLogit = lastLogit;
  let fitSigma = NO_FIT_SIGMA;
  if (fit !== null) {
    const trendLogit = fit.intercept + fit.slope * ctx.xTarget;
    centreLogit = blendCentre(lastLogit, trendLogit, MAX_TREND_SHARE * fit.rSquared);
    const leverage = 1 / fit.count + ((ctx.xTarget - fit.meanX) ** 2) / fit.sumSqX;
    fitSigma = Math.min(FIT_SIGMA_CAP, fit.residualStd * Math.sqrt(Math.max(0, leverage)));
  }

  const sigma = Math.sqrt(
    (WALK_SIGMA * ctx.sqrtDays) ** 2 +
      fitSigma ** 2 +
      (REGIME_BASE + REGIME_PER_DAY * ctx.sqrtDays) ** 2,
  );
  const q = ctx.z.map((zz) => round4dp(clampTo(fromLogit(centreLogit + zz * sigma), 0, 100)));
  // Rounding must not break monotonicity.
  for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
  return { id: trophy.id, q };
};

/** Forecast every trophy of one case. */
const forecastCase = (c: CaseFile): { trophies: Array<{ id: number; q: number[] }> } => {
  const levels = c.levels.length > 0 ? c.levels : QUANTILE_LEVELS;
  const standardLevels =
    levels.length === QUANTILE_LEVELS.length && levels.every((t, j) => t === QUANTILE_LEVELS[j]);
  const z = standardLevels ? NORMAL_Z : levels.map((t) => normInv(clampTo(t, 0.0001, 0.9999)));

  const checkpointMs = Date.parse(c.at);
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const sqrtDays = Math.sqrt(horizonDays);
  const targetMs = Number.isFinite(checkpointMs)
    ? checkpointMs + Math.max(0, c.horizon_minutes) * 60_000
    : EPOCH_MS;
  const xTarget = logHours(targetMs);
  const windowStart = Number.isFinite(checkpointMs) ? checkpointMs - LOOKBACK_MS : -Infinity;

  const ctx: CaseContext = { z, xTarget, windowStart, sqrtDays };
  return { trophies: c.trophies.map((t, i) => forecastTrophy(t, i, c, ctx)) };
};

mkdirSync(OUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(IN_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(IN_DIR, file), 'utf8')) as CaseFile;
  writeFileSync(join(OUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
