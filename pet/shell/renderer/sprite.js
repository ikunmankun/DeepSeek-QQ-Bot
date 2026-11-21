// 大肥鱼 2D 精灵渲染层：DOM 图片 + 程序化动画（呼吸/漂浮/摇摆/挤压/转圈/小跳）。
// 命中检测用像素 alpha（离屏 canvas），窗口空角保持点击穿透。
const body = document.getElementById('body');
const bubble = document.getElementById('bubble');

body.src = '../../sprite/body.png?v=' + Date.now();  // 换图时绕过缓存

// ---------- 状态 ----------
const state = {
  patT: -1, spinT: -1, hopT: -1, danceT: -1,
  danceNext: 40 + Math.random() * 80,
  bobPhase: Math.random() * 6,
  dragging: false,
  leanTarget: 0, lean: 0,        // 视线/身体倾向鼠标
};

// ---------- 台词 ----------
const IDLE_LINES = [
  '干啥呢，饭点都不叫我',
  '这活费token，加饭',
  '别戳了，气鼓鱼也是鱼',
  '今天的记忆星图亮不亮？',
  '夜宵考虑一下白饭？',
  '2D 的我，精致多了吧',
  '谁又发表情包了，我看看',
  '摸鱼可以，摸我请排队',
];
const PAT_LINES = ['哼，手挺凉', '再戳就鼓成球了', '行吧，准你摸这一下', '鱼离不开水，我离不开饭', '哦。'];
const HAPPY_LINES = ['好耶！', 'token 到账，开炫', '今天也是被需要的一天'];

let bubbleTimer = null;
function bubbleSay(text, ms = 4200) {
  bubble.textContent = text;
  bubble.classList.add('show');
  state.hopT = 0;
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => bubble.classList.remove('show'), ms);
}
const pick = (list) => list[Math.floor(Math.random() * list.length)];

function isNight(now = new Date()) {
  const h = now.getHours();
  return h >= 23 || h < 8;
}

// ---------- 像素级命中：离屏 canvas 读 alpha ----------
let hitCanvas = null;
body.addEventListener('load', () => {
  hitCanvas = document.createElement('canvas');
  hitCanvas.width = body.naturalWidth;
  hitCanvas.height = body.naturalHeight;
  const ctx = hitCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(body, 0, 0);
  console.log('[pet] sprite loaded', body.naturalWidth, 'x', body.naturalHeight);
});
body.addEventListener('error', () => console.log('[pet] sprite load FAILED'));

function overWhale(clientX, clientY) {
  if (!hitCanvas) return true;   // 图没加载完时按窗口处理
  const rect = body.getBoundingClientRect();
  if (clientX < rect.left || clientX >= rect.right
      || clientY < rect.top || clientY >= rect.bottom) return false;
  // 显示坐标 → 图片像素坐标（忽略 ±2° 的微小旋转）
  const ix = Math.floor((clientX - rect.left) / rect.width * hitCanvas.width);
  const iy = Math.floor((clientY - rect.top) / rect.height * hitCanvas.height);
  try {
    return hitCanvas.getContext('2d', { willReadFrequently: true })
      .getImageData(ix, iy, 1, 1).data[3] > 24;
  } catch (_) { return true; }
}

// ---------- 交互（与 3D 版同一套 IPC） ----------
let pressPoint = null;
let lastInteractive = null;
window.addEventListener('mousemove', (e) => {
  const over = overWhale(e.clientX, e.clientY);
  if (over !== lastInteractive) {
    lastInteractive = over;
    window.petAPI.setOverWhale(over);
    body.style.cursor = over ? 'grab' : 'default';
  }
  state.leanTarget = ((e.clientX / window.innerWidth) - 0.5) * 6;  // 度，最多±3°
  if (state.dragging && pressPoint) window.petAPI.dragMove(e.screenX, e.screenY);
});

window.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || !overWhale(e.clientX, e.clientY)) return;
  pressPoint = { x: e.clientX, y: e.clientY, time: Date.now() };
  state.dragging = true;
  window.petAPI.dragStart(e.screenX - e.clientX, e.screenY - e.clientY);
});

window.addEventListener('mouseup', (e) => {
  if (state.dragging) { state.dragging = false; window.petAPI.dragEnd(); }
  if (pressPoint && overWhale(e.clientX, e.clientY)) {
    const moved = Math.hypot(e.clientX - pressPoint.x, e.clientY - pressPoint.y);
    if (moved < 4 && Date.now() - pressPoint.time < 500) {
      state.patT = 0;
      bubbleSay(pick(PAT_LINES));
    }
  }
  pressPoint = null;
});

window.addEventListener('dblclick', (e) => {
  if (overWhale(e.clientX, e.clientY)) { state.spinT = 0; bubbleSay(pick(HAPPY_LINES)); }
});

window.addEventListener('contextmenu', (e) => {
  if (overWhale(e.clientX, e.clientY)) { e.preventDefault(); window.petAPI.showMenu(); }
});

// 滚轮缩放（底边锚定，尺寸写回配置）
window.addEventListener('wheel', (e) => {
  if (!overWhale(e.clientX, e.clientY)) return;
  e.preventDefault();
  sizeScale = Math.min(2.2, Math.max(0.6, sizeScale * (e.deltaY < 0 ? 1.07 : 1 / 1.07)));
  window.petAPI.resizeWindow(sizeScale);
}, { passive: false });

// ---------- 尺寸配置 ----------
let sizeScale = 1;
(async () => {
  try {
    const config = await window.petAPI.getConfig();
    const saved = Math.min(2.2, Math.max(0.6, Number(config && config.sizeScale) || 1));
    if (Math.abs(saved - 1) > 0.01) { sizeScale = saved; window.petAPI.resizeWindow(sizeScale); }
  } catch (_) {}
})();
window.petAPI.onResetSize(() => { sizeScale = 1; window.petAPI.resizeWindow(1); bubbleSay('恢复默认体型'); });

// ---------- 主动冒泡 ----------
setInterval(() => { if (!isNight()) bubbleSay(pick(IDLE_LINES)); }, 6 * 60 * 1000);
setTimeout(() => { if (!isNight()) bubbleSay(IDLE_LINES[0]); }, 15000);
async function memoryReport() {
  try {
    const res = await fetch('http://127.0.0.1:8765/api/memory-map', { cache: 'no-store' });
    const data = await res.json();
    if (data.available && data.stats && data.stats.total) {
      bubbleSay(`现在记得 ${data.stats.total} 件事，` +
        `${data.stats.immutable} 件永不遗忘，${data.stats.faded} 件在淡忘`);
    }
  } catch (_) {}
}
setInterval(memoryReport, 10 * 60 * 1000);
setTimeout(memoryReport, 8000);

// ---------- 主循环 ----------
const clock = { last: performance.now() };
let lastT = 0;
function animate(nowMs) {
  requestAnimationFrame(animate);
  const dt = Math.min((nowMs - clock.last) / 1000, 0.05);
  clock.last = nowMs;
  const t = nowMs / 1000;
  const night = isNight();
  body.classList.toggle('night', night);

  const bobSpeed = night ? 4.5 : 11;                 // rad/s，越慢越困
  const bobAmp = night ? 2 : 6;                      // px
  const breathe = 1 + Math.sin(t * bobSpeed * 0.35 + state.bobPhase) * (night ? 0.006 : 0.012);

  let squashY = 1;
  if (state.patT >= 0) {
    state.patT += dt;
    const decay = Math.exp(-state.patT * 5.2);
    squashY = 1 - 0.14 * Math.sin(state.patT * 17) * decay;
    if (state.patT > 0.9) state.patT = -1;
  }

  let hop = 0;
  if (state.hopT >= 0) {
    state.hopT += dt;
    hop = Math.sin(Math.min(1, state.hopT / 0.5) * Math.PI) * 10;
    if (state.hopT >= 0.5) state.hopT = -1;
  }

  let danceRot = 0, danceHop = 0;
  state.danceNext -= dt;
  if (state.danceNext <= 0 && !night && !state.dragging) { state.danceT = 0; state.danceNext = 120 + Math.random() * 180; }
  if (state.danceT >= 0) {
    state.danceT += dt;
    danceRot = Math.sin(state.danceT * 10) * 5;
    danceHop = Math.abs(Math.sin(state.danceT * 8)) * 8;
    if (state.danceT >= 2.4) state.danceT = -1;
  }

  state.lean += (state.leanTarget - state.lean) * Math.min(1, dt * 5);

  const bob = Math.sin(t * bobSpeed * 0.5 + state.bobPhase) * bobAmp;
  const sway = Math.sin(t * 0.5) * 1.5;
  let rot = sway + state.lean;
  let scaleX = breathe, scaleY = breathe;
  if (state.patT >= 0) { scaleY *= squashY; scaleX *= 2 - squashY; }

  let flip = '';
  if (state.spinT >= 0) {
    state.spinT += dt;
    const p = Math.min(1, state.spinT / 0.9);
    flip = ` rotateY(${(p * 360).toFixed(1)}deg)`;
    if (state.spinT >= 0.9) state.spinT = -1;
  }

  body.style.transform =
    `translate(${(-state.lean * 1.2).toFixed(1)}px, ${(bob + hop + danceHop).toFixed(1)}px) ` +
    `rotate(${(rot + danceRot).toFixed(2)}deg) scale(${scaleX.toFixed(3)}, ${scaleY.toFixed(3)})` + flip;
}
requestAnimationFrame(animate);
