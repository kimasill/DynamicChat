/**
 * Night-theme regression check.
 *
 * The night theme works by redefining tokens in `:root[data-reader-theme="night"]`
 * (src/narrative-output.css, imported after styles.css so it wins on source order). Any rule that
 * hard-codes a colour literal is invisible to that mechanism: when its counterpart half DOES follow the
 * theme, the pair drifts apart and you get light-on-light or dark-on-dark.
 *
 * This reads the stylesheets rather than a rendered page, because a browser can only measure the view
 * that happens to be open and most of this app is three clicks deep. It reports a declaration only when
 * it is LIVE (no later rule with the same selector part re-sets that property) and NOT already covered by
 * a night rule.
 *
 * Three things it took a while to get right, all of which caused false results before:
 *   - the cascade must be resolved per SELECTOR PART, not per selector text: `.a` and `.a, .b` are the
 *     same rule for `.a`, and keying on the whole text reported declarations that a later comma-list
 *     already overrode
 *   - `:is()` has to be expanded with brace matching, not a regex — `:is(a, b:not(.c))` contains parens,
 *     and a failed expansion silently reports a covered selector as uncovered
 *   - CSS attribute VALUES must survive parsing; strip the quotes and every night rule becomes invisible
 *
 * ALLOWED holds the declarations that look like defects and are not, each with the reason. Adding to it
 * is a deliberate act: state why the pair is correct, or fix the CSS.
 *
 *   node server/theme-contrast-eval.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FILES = ["src/styles.css", "src/narrative-output.css"];

/* Declarations that are flagged by the heuristics but are correct. */
const ALLOWED = [
  {
    match: (f) => f.part === ".vibe-encode-state.stale",
    why: "#b8860b already measures 4.98:1 on --dc-panel. Restating it would imply a decision nobody made."
  },
  {
    match: (f) => f.part.startsWith(".crack-send-button.is-stop"),
    why: "Red fill with white ink — a fixed pair that is dark in both themes. Flipping either half breaks it."
  },
  {
    match: (f) => f.part.startsWith(".image-asset-delete:"),
    why: "Same fixed pair: #b8503e fill under white ink, dark in both themes."
  },
  {
    match: (f) => /^\.(episode-art|scene-poster-window|relationship-node-portrait)$|^\.scene-visual::after$/.test(f.part),
    why: "A white edge drawn over artwork whose colour does not follow the theme. The edge belongs to the picture."
  }
];

const LIGHT = 0.55;
const DARK = 0.35;

/* ---------- colour ---------- */
function parseColor(value) {
  const v = String(value).trim();
  let m = v.match(/^#([0-9a-f]{3,8})$/i);
  if (m) {
    let h = m[1];
    if (h.length === 3 || h.length === 4) h = h.split("").map((c) => c + c).join("");
    return {
      r: parseInt(h.slice(0, 2), 16),
      g: parseInt(h.slice(2, 4), 16),
      b: parseInt(h.slice(4, 6), 16),
      a: h.length >= 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1
    };
  }
  m = v.match(/^rgba?\(([^)]+)\)$/i);
  if (m) {
    const p = m[1].split(/[,\s/]+/).filter(Boolean);
    if (p.length < 3) return null;
    const num = (s) => (s.endsWith("%") ? parseFloat(s) * 2.55 : parseFloat(s));
    const a = p[3] !== undefined ? (p[3].endsWith("%") ? parseFloat(p[3]) / 100 : parseFloat(p[3])) : 1;
    return { r: num(p[0]), g: num(p[1]), b: num(p[2]), a };
  }
  if (/^white$/i.test(v)) return { r: 255, g: 255, b: 255, a: 1 };
  if (/^black$/i.test(v)) return { r: 0, g: 0, b: 0, a: 1 };
  return null;
}
const luminance = ({ r, g, b }) => {
  const f = (v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const firstLiteral = (v) => {
  const m = String(v).match(/#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)/);
  return m ? parseColor(m[0]) : null;
};
const usesVar = (v) => /var\(\s*--/.test(String(v));

/* ---------- CSS ---------- */
const BS = String.fromCharCode(92);
function skipString(s, start) {
  const q = s[start];
  let j = start + 1;
  while (j < s.length && s[j] !== q) {
    if (s[j] === BS) j++;
    j++;
  }
  return j + 1;
}

function parseCss(src) {
  const rules = [];
  const newlines = [];
  for (let k = 0; k < src.length; k++) if (src[k] === "\n") newlines.push(k);
  const lineAt = (idx) => {
    let lo = 0, hi = newlines.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (newlines[mid] < idx) lo = mid + 1;
      else hi = mid;
    }
    return lo + 1;
  };
  const atStack = [];
  let i = 0, depth = 0, buf = "";
  while (i < src.length) {
    const ch = src[i];
    if (ch === "/" && src[i + 1] === "*") {
      const e = src.indexOf("*/", i + 2);
      i = e === -1 ? src.length : e + 2;
      continue;
    }
    if (ch === '"' || ch === "'") { i = skipString(src, i); continue; }
    if (ch === "{") {
      const prelude = buf.trim();
      if (prelude.startsWith("@")) { atStack.push({ depth, prelude }); depth++; buf = ""; i++; continue; }
      let j = i + 1, d = 1;
      while (j < src.length && d > 0) {
        const c = src[j];
        if (c === "/" && src[j + 1] === "*") { const e = src.indexOf("*/", j + 2); j = e === -1 ? src.length : e + 2; continue; }
        if (c === '"' || c === "'") { j = skipString(src, j); continue; }
        if (c === "{") d++;
        else if (c === "}") d--;
        j++;
      }
      // Walk back over the raw selector so a comment above the rule is not swallowed, then re-read the
      // selector from the SOURCE: the scanner skips quoted strings, so `[data-reader-theme="night"]`
      // would otherwise arrive as `[data-reader-theme=]` and no night rule would ever be recognised.
      let s = i - 1;
      while (s >= 0) {
        const c = src[s];
        if (c === "}" || c === ";") break;
        if (c === "/" && src[s - 1] === "*") break;
        s--;
      }
      let start = s + 1;
      while (start < i && /[ \t\r\n]/.test(src[start])) start++;
      rules.push({
        selector: src.slice(start, i).replace(/\/\*[\s\S]*?\*\//g, " ").replace(/[ \t\r\n]+/g, " ").trim(),
        body: src.slice(i + 1, j - 1),
        inAt: atStack.length ? atStack.map((a) => a.prelude).join(" / ") : null,
        line: lineAt(start)
      });
      i = j; buf = ""; continue;
    }
    if (ch === "}") {
      depth--;
      if (atStack.length && atStack[atStack.length - 1].depth === depth) atStack.pop();
      buf = ""; i++; continue;
    }
    buf += ch; i++;
  }
  return rules;
}

function declarationsOf(body) {
  const out = [];
  let i = 0, buf = "", depth = 0;
  while (i < body.length) {
    const ch = body[i];
    if (ch === "/" && body[i + 1] === "*") { const e = body.indexOf("*/", i + 2); i = e === -1 ? body.length : e + 2; continue; }
    if (ch === '"' || ch === "'") {
      const q = ch; buf += ch; i++;
      while (i < body.length && body[i] !== q) { if (body[i] === BS) { buf += body[i]; i++; } buf += body[i]; i++; }
      buf += body[i] || ""; i++; continue;
    }
    if (ch === "(") depth++;
    else if (ch === ")") depth--;
    if (ch === ";" && depth === 0) { if (buf.trim()) out.push(buf.trim()); buf = ""; i++; continue; }
    buf += ch; i++;
  }
  if (buf.trim()) out.push(buf.trim());
  return out
    .map((d) => {
      const c = d.indexOf(":");
      if (c === -1) return null;
      const prop = d.slice(0, c).trim();
      if (!prop || /[{}]/.test(prop)) return null;
      return { prop, value: d.slice(c + 1).trim() };
    })
    .filter(Boolean);
}

function splitSelectorList(sel) {
  const out = [];
  let d = 0, cur = "";
  for (const ch of sel) {
    if (ch === "(") d++;
    else if (ch === ")") d--;
    if (ch === "," && d === 0) { out.push(cur.trim()); cur = ""; continue; }
    cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function expandIs(sel) {
  const at = sel.indexOf(":is(");
  if (at === -1) return [sel];
  let i = at + 4, d = 1;
  while (i < sel.length && d > 0) { if (sel[i] === "(") d++; else if (sel[i] === ")") d--; i++; }
  const inner = sel.slice(at + 4, i - 1);
  const out = [];
  for (const alt of splitSelectorList(inner)) out.push(...expandIs(sel.slice(0, at) + alt + sel.slice(i)));
  return out;
}

function specificity(sel) {
  const s = sel.replace(/::[\w-]+/g, " E ");
  return [
    (s.match(/#[\w-]+/g) || []).length,
    (s.match(/\.[\w-]+|\[[^\]]*\]|:(?!:)[\w-]+(\([^)]*\))?/g) || []).length,
    (s.match(/(^|[\s>+~])[a-zA-Z][\w-]*/g) || []).length
  ];
}
const compareSpecificity = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

/* ---------- analysis ---------- */
const rules = [];
for (const rel of FILES) {
  const src = fs.readFileSync(path.join(root, rel), "utf8");
  for (const rule of parseCss(src)) {
    rule.file = rel;
    rule.decls = declarationsOf(rule.body);
    rules.push(rule);
  }
}

const NIGHT = /\[data-reader-theme="night"\]/;
const nightTargets = [];
for (const rule of rules) {
  if (!NIGHT.test(rule.selector)) continue;
  for (const part of splitSelectorList(rule.selector)) {
    for (const expanded of expandIs(part)) {
      nightTargets.push({
        sel: expanded.trim(),
        spec: specificity(expanded),
        props: new Set(rule.decls.map((d) => d.prop))
      });
    }
  }
}

function nightCovers(lightPart, prop) {
  const lp = lightPart.trim();
  const lspec = specificity(lp);
  for (const nt of nightTargets) {
    const handlesProp =
      nt.props.has(prop) ||
      (prop.startsWith("border") && [...nt.props].some((p) => p.startsWith("border")));
    if (!handlesProp) continue;
    const tail = nt.sel
      .replace(/^:root\[data-reader-theme="night"\]\s*/, "")
      .replace(/^\.crack-story-stage\[data-reader-theme="night"\]\s*/, "")
      .trim();
    if (!tail) continue;
    if (tail === lp || tail.endsWith(" " + lp) || lp.endsWith(tail) || tail.startsWith(lp)) {
      if (compareSpecificity(nt.spec, lspec) >= 0) return true;
    }
  }
  return false;
}

/* Live cascade, keyed per selector PART. */
const byPart = new Map();
for (const rule of rules) {
  if (NIGHT.test(rule.selector)) continue;
  for (const part of splitSelectorList(rule.selector)) {
    for (const expanded of expandIs(part)) {
      const key = (rule.inAt || "") + " || " + expanded.trim();
      if (!byPart.has(key)) byPart.set(key, new Map());
      const props = byPart.get(key);
      for (const d of rule.decls) {
        if (!props.has(d.prop)) props.set(d.prop, []);
        props.get(d.prop).push({ rule, decl: d });
      }
    }
  }
}

const findings = [];
for (const [key, props] of byPart) {
  const part = key.split(" || ")[1];
  const winner = new Map();
  for (const [prop, list] of props) winner.set(prop, list[list.length - 1]);

  const bgEntry = winner.get("background-color") || winner.get("background");
  const colorEntry = winner.get("color");
  const bg = bgEntry?.decl.value;
  const color = colorEntry?.decl.value;
  const bgLiteral = bg && !usesVar(bg) ? firstLiteral(bg) : null;
  const colorLiteral = color && !usesVar(color) ? parseColor(color) : null;

  const add = (prop, kind, detail, entry) => {
    if (nightCovers(part, prop)) return;
    findings.push({ kind, part, prop, detail, file: entry.rule.file, line: entry.rule.line, value: entry.decl.value });
  };

  if (bgLiteral && bgLiteral.a > 0.5 && luminance(bgLiteral) > LIGHT) {
    if (color && usesVar(color)) add("background", "pair-split", `light literal fill under themed colour ${color}`, bgEntry);
    else if (!color) add("background", "light-surface", "light literal fill, colour inherited from the themed ink", bgEntry);
  }
  if (colorLiteral && colorLiteral.a > 0.5 && luminance(colorLiteral) < DARK) {
    if (bg && usesVar(bg)) add("color", "pair-split", `dark literal ink on themed fill ${bg}`, colorEntry);
    else if (!bg) add("color", "dark-ink", "dark literal ink sitting on the themed surface", colorEntry);
  }
  for (const [prop, entry] of winner) {
    if (!/^border(-(top|right|bottom|left))?(-color)?$|^outline(-color)?$/.test(prop)) continue;
    if (usesVar(entry.decl.value)) continue;
    const lit = firstLiteral(entry.decl.value);
    if (!lit || lit.a < 0.35) continue;
    const l = luminance(lit);
    if (l > 0.5 || l < 0.12) add(prop, "border-literal", `literal ${prop} on a surface that flips`, entry);
  }
}

const allowed = [];
const open = [];
for (const f of findings) {
  const rule = ALLOWED.find((a) => a.match(f));
  if (rule) allowed.push({ ...f, why: rule.why });
  else open.push(f);
}

console.log(`야간 테마 대비 검사 — 규칙 ${rules.length}개, 야간 오버라이드 ${nightTargets.length}개`);
console.log("");
for (const f of open) {
  console.log(`FAIL ${f.file}:${f.line}  ${f.prop}  ${f.part}`);
  console.log(`     ${f.value}`);
  console.log(`     ${f.detail}`);
}
if (allowed.length) {
  console.log(`허용된 예외 ${allowed.length}건:`);
  const seen = new Set();
  for (const f of allowed) {
    if (seen.has(f.why)) continue;
    seen.add(f.why);
    console.log(`  · ${f.why}`);
  }
  console.log("");
}
console.log(`Theme contrast checks: ${findings.length - open.length} allowed, ${open.length} failures`);
if (open.length > 0) process.exitCode = 1;
