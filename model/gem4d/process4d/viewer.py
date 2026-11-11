"""4D-process outputs: a self-contained interactive 3D viewer (HTML, three.js) in the style of
DynPoint's Stage-1 viewer, a 2D summary video (frame | depth & scene-flow arrows) and the raw arrays.

Viewer: per selected frame t, the frame's point cloud (RGB or depth colours), yellow->red segments
for the scene flow of dynamic points to frame t+gap, the dynamic points at t+gap in blue, and the
camera frustum of frame t in blue. Controls: frame slider / play, colour mode, flow length, point
size, layer toggles; several sequences can share one page.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scene_flow import Result4D

DISPLAY = np.diag([1.0, -1.0, -1.0])        # OpenCV (x right, y down, z forward) -> three.js (y up)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _jpeg(img: np.ndarray, quality=85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def depth_to_rgb(depth: np.ndarray, lo, hi) -> np.ndarray:
    x = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    # turbo-like ramp, near = red, far = blue
    stops = np.array([[0.19, 0.07, 0.23], [0.16, 0.47, 0.93], [0.11, 0.81, 0.69], [0.64, 0.99, 0.24],
                      [0.98, 0.73, 0.21], [0.88, 0.27, 0.05], [0.48, 0.02, 0.01]])[::-1]
    pos = x * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    t = (pos - i)[..., None]
    return ((stops[i] * (1 - t) + stops[i + 1] * t) * 255).astype(np.uint8)


def summary_frames(r: Result4D) -> np.ndarray:
    """[F-gap, H, 2W, 3]: frame | depth with scene-flow arrows on the moving foreground (flow_vis.FlowPanel;
    camera shake removed, no arrows on the background)."""
    from .flow_vis import FlowPanel

    panel = FlowPanel.from_result(r)
    return np.stack([np.concatenate([r.images[t], panel(t)], 1) for t in range(panel.n)])


def viewer_sequence(r: Result4D, name: str, frame_step: int = 4, max_points: int = 20000,
                    max_dynamic: int = 6000, seed: int = 0) -> dict:
    """Subsample one Result4D into the viewer's JSON payload (scene normalised to unit median radius)."""
    rng = np.random.default_rng(seed)
    n_flow = r.sf.shape[0]
    center = np.median(r.points[0][r.conf_ok[0]], axis=0)
    scale = 1.0 / max(np.median(np.linalg.norm(r.points[0][r.conf_ok[0]] - center, axis=-1)), 1e-6)
    to_disp = lambda p: ((p - center) * scale) @ DISPLAY.T                       # noqa: E731
    dlo, dhi = np.percentile(r.depth, [2, 98])
    h, w = r.depth.shape[1:]
    frames = []
    for t in range(0, n_flow, frame_step):
        ok = np.flatnonzero(r.conf_ok[t].ravel())
        idx = rng.choice(ok, min(max_points, ok.size), replace=False) if ok.size else ok
        dyn = np.flatnonzero(r.dynamic[t].ravel())
        didx = rng.choice(dyn, min(max_dynamic, dyn.size), replace=False) if dyn.size else dyn
        pts = r.points[t].reshape(-1, 3)
        sf = r.sf[t].reshape(-1, 3)
        c2w = r.c2w[t].copy()
        c2w_disp = np.eye(4)
        c2w_disp[:3, :3] = DISPLAY @ c2w[:3, :3] @ DISPLAY.T
        c2w_disp[:3, 3] = DISPLAY @ ((c2w[:3, 3] - center) * scale)
        d8 = np.clip((r.depth[t].ravel()[idx] - dlo) / max(dhi - dlo, 1e-6) * 255, 0, 255).astype(np.uint8)
        mag = np.linalg.norm(r.sf[t][r.dynamic[t]], axis=-1) * scale
        frames.append(dict(
            t=int(t), pos=_b64(to_disp(pts[idx]).astype(np.float32)),
            rgb=_b64(r.images[t].reshape(-1, 3)[idx].astype(np.uint8)), depth=_b64(d8),
            dyn0=_b64(to_disp(pts[didx]).astype(np.float32)),
            dyn1=_b64(to_disp(pts[didx] + sf[didx]).astype(np.float32)),
            n=int(idx.size), m=int(didx.size), dyn_total=int(dyn.size),
            dyn_mean=float(mag.mean()) if mag.size else 0.0,
            c2w=c2w_disp.T.ravel().tolist(),                                        # column-major for three.js
            K=r.K[t].ravel().tolist(),
            img=_jpeg(np.concatenate([r.images[t], r.images[t + r.gap]], 1)),
        ))
    return dict(name=name, width=int(w), height=int(h), gap=int(r.gap), num_frames=int(r.depth.shape[0]),
                stats=r.stats(), frames=frames)


def write_viewer(path, sequences: list, title: str = "GEM-4D 4D-process"):
    """Write one self-contained HTML viewer for a list of viewer_sequence() payloads."""
    html = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(sequences))
    Path(path).write_text(html)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mut:#6b6a65;--line:#e2dfd7;--acc:#2f6fdf;--canvas:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16161a;--card:#202024;--ink:#ecebe6;--mut:#a09f99;--line:#34343a;--acc:#6f9cf0;--canvas:#101014}}
:root[data-theme="dark"]{--bg:#16161a;--card:#202024;--ink:#ecebe6;--mut:#a09f99;--line:#34343a;--acc:#6f9cf0;--canvas:#101014}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:12px 16px}
.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:8px 0}
.tabs button,.bar button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
label{color:var(--mut);display:inline-flex;gap:6px;align-items:center}
#view{position:relative;width:100%;aspect-ratio:16/9;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--canvas)}
#view canvas{display:block;width:100%;height:100%}
#hint{position:absolute;left:10px;bottom:8px;color:var(--mut);font-size:12px;pointer-events:none}
.row{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:12px;margin-top:10px}
@media (max-width:760px){.row{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.card img{width:100%;display:block;border-radius:6px}
.mut{color:var(--mut);font-size:12.5px}.kv{font-variant-numeric:tabular-nums}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body><div class="wrap">
<div class="tabs bar" id="tabs"></div>
<div id="view"><div id="hint">drag: rotate · right-drag: pan · scroll: zoom</div></div>
<div class="bar">
 <button id="play">▶ play</button>
 <label>frame <input id="frame" type="range" min="0" max="0" value="0"> <span id="flab" class="kv"></span></label>
 <label>colour <select id="color"><option value="rgb">RGB</option><option value="depth">depth</option></select></label>
 <label>flow length <input id="flen" type="range" min="0.5" max="5" step="0.1" value="1"> <span id="flv" class="kv">1.0×</span></label>
 <label>point size <input id="psize" type="range" min="1" max="6" step="0.5" value="2"></label>
 <label><input id="sflow" type="checkbox" checked> flow</label>
 <label><input id="starget" type="checkbox" checked> target</label>
 <label><input id="scam" type="checkbox" checked> camera</label>
</div>
<div class="row">
 <div class="card"><img id="img" alt="frame t and t+gap"><div class="mut" id="imgcap"></div></div>
 <div class="card"><div class="legend mut">
  <div><span style="background:linear-gradient(90deg,#ffd400,#e3170a)"></span>scene flow t → t+gap (dynamic points)</div>
  <div><span style="background:#2f6fdf"></span>dynamic points at t+gap · camera of frame t</div></div>
  <div id="stats" class="mut kv" style="margin-top:8px"></div></div>
</div></div>
<script>window.P4D_DATA = __DATA__;</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA = window.P4D_DATA, $ = id => document.getElementById(id);
const f32 = s => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new Float32Array(b.buffer); };
const u8 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
const view = $('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const bg = () => getComputedStyle(document.documentElement).getPropertyValue('--canvas').trim() || '#fff';
const TURBO = [[.48,.02,.01],[.88,.27,.05],[.98,.73,.21],[.64,.99,.24],[.11,.81,.69],[.16,.47,.93],[.19,.07,.23]];
function ramp(x) { const p = x * (TURBO.length - 1), i = Math.min(Math.floor(p), TURBO.length - 2), t = p - i;
  return TURBO[i].map((a, k) => a * (1 - t) + TURBO[i + 1][k] * t); }
let seq = null, cache = [], groups = {}, cur = 0, timer = null;
function decode(fr) {
  const n = fr.n, pos = f32(fr.pos), rgb = u8(fr.rgb), dep = u8(fr.depth);
  const cRGB = new Float32Array(n * 3), cD = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { for (let k = 0; k < 3; k++) cRGB[3*i+k] = rgb[3*i+k] / 255;
    const c = ramp(dep[i] / 255); cD[3*i] = c[0]; cD[3*i+1] = c[1]; cD[3*i+2] = c[2]; }
  return {pos, cRGB, cD, d0: f32(fr.dyn0), d1: f32(fr.dyn1), m: fr.m};
}
function frustum(fr, size) {
  const K = fr.K, w = seq.width, h = seq.height, fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  const cs = [[0,0],[w,0],[w,h],[0,h]].map(([u,v]) => new THREE.Vector3((u-cx)/fx*size, -(v-cy)/fy*size, -size));
  const o = new THREE.Vector3(), p = [];
  for (const c of cs) p.push(o, c);
  for (let i = 0; i < 4; i++) p.push(cs[i], cs[(i+1)%4]);
  const g = new THREE.BufferGeometry().setFromPoints(p);
  const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({color: 0x2f6fdf}));
  l.matrixAutoUpdate = false; l.matrix.fromArray(fr.c2w); return l;
}
function clear() { for (const k in groups) { scene.remove(groups[k]); groups[k].geometry?.dispose(); } groups = {}; }
function draw() {
  if (!seq) return;
  clear();
  const fr = seq.frames[cur], d = cache[cur] || (cache[cur] = decode(fr));
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(d.pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute($('color').value === 'rgb' ? d.cRGB : d.cD, 3));
  groups.cloud = new THREE.Points(g, new THREE.PointsMaterial({size: +$('psize').value, sizeAttenuation: false, vertexColors: true}));
  scene.add(groups.cloud);
  const L = +$('flen').value, m = d.m;
  if ($('sflow').checked && m) {
    const p = new Float32Array(m * 6), c = new Float32Array(m * 6);
    for (let i = 0; i < m; i++) for (let k = 0; k < 3; k++) {
      p[6*i+k] = d.d0[3*i+k]; p[6*i+3+k] = d.d0[3*i+k] + L * (d.d1[3*i+k] - d.d0[3*i+k]); }
    for (let i = 0; i < m; i++) { c.set([1, .83, 0], 6*i); c.set([.89, .09, .04], 6*i+3); }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(p, 3)); lg.setAttribute('color', new THREE.BufferAttribute(c, 3));
    groups.flow = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({vertexColors: true}));
    scene.add(groups.flow);
  }
  if ($('starget').checked && m) {
    const tg = new THREE.BufferGeometry(); tg.setAttribute('position', new THREE.BufferAttribute(d.d1, 3));
    groups.target = new THREE.Points(tg, new THREE.PointsMaterial({size: +$('psize').value + 1, sizeAttenuation: false, color: 0x2f6fdf}));
    scene.add(groups.target);
  }
  if ($('scam').checked) { groups.cam = frustum(fr, 0.25); scene.add(groups.cam); }
  $('flab').textContent = `t = ${fr.t} → ${fr.t + seq.gap}`;
  $('img').src = 'data:image/jpeg;base64,' + fr.img;
  $('imgcap').textContent = `frame ${fr.t} (left) and frame ${fr.t + seq.gap} (right), ${seq.width}×${seq.height} DA3 resolution`;
  const s = seq.stats;
  $('stats').innerHTML = `dynamic pixels in this frame: ${fr.dyn_total} (${fr.m} shown)<br>mean |scene flow| (dynamic, scene units): ${fr.dyn_mean.toFixed(3)}<br>`
    + `video: ${s.frames} frames, gap ${s.gap}, valid ${(100*s.valid_fraction).toFixed(1)}%, dynamic ${(100*s.dynamic_fraction).toFixed(2)}%<br>`
    + `static residual / dynamic mean flow: ${(s.static_median_flow/s.median_depth).toFixed(4)} / ${(s.dynamic_mean_flow/s.median_depth).toFixed(4)} × median depth`;
}
function setSeq(i) {
  seq = DATA[i]; cache = []; cur = 0;
  document.querySelectorAll('#tabs button').forEach((b, k) => b.classList.toggle('on', k === i));
  $('frame').max = seq.frames.length - 1; $('frame').value = 0;
  const c = new THREE.Matrix4().fromArray(seq.frames[0].c2w);
  const back = new THREE.Vector3(0, 0.25, 0.9).applyMatrix4(c);
  camera.position.copy(back); controls.target.set(0, 0, 0); controls.update();   // scene is centred on its median point
  draw();
}
DATA.forEach((s, i) => { const b = document.createElement('button'); b.textContent = s.name; b.onclick = () => setSeq(i); $('tabs').append(b); });
if (DATA.length < 2) $('tabs').style.display = 'none';
$('frame').oninput = e => { cur = +e.target.value; draw(); };
for (const id of ['color', 'psize', 'sflow', 'starget', 'scam']) $(id).oninput = draw;
$('flen').oninput = () => { $('flv').textContent = (+$('flen').value).toFixed(1) + '×'; draw(); };
$('play').onclick = () => { if (timer) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
  $('play').textContent = '❚❚ pause';
  timer = setInterval(() => { cur = (cur + 1) % seq.frames.length; $('frame').value = cur; draw(); }, 600); };
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(view); resize();
(function loop() { scene.background = new THREE.Color(bg()); controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); })();
setSeq(0);
</script></body></html>
"""
