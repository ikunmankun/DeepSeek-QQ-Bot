// 大肥鱼 webm 动画渲染层：播放 pet/anim/webm 的透明动画池。
// 节奏核心：每段播完原地停留 restSeconds 再掷下一段（权重见 pools.json），
// 待机占大头，桌宠整体"安静但有活气"。命中检测用当前帧的像素 alpha。
const video = document.getElementById('video');
const bubble = document.getElementById('bubble');

const config = await window.petAPI.getAnimConfig();
if (!config) {
  console.log('[pet] pools.json missing; idle forever');
}

const state = {
  facing: 'left',          // 东张西望播完翻转
  dragging: false,
  sleeping: false,         // 桥探测到 QQ 掉线时的睡觉态
  current: '',
  restTimer: null,
  missing: new Set(),      // 播不出来的动画名，本轮会话不再选它
};

// ---------- 基础 ----------
const pick = (list) => list[Math.floor(Math.random() * list.length)];

function usable(list) {
  const alive = (list || []).filter((name) => !state.missing.has(name));
  return alive.length ? alive : (list || []);
}

function play(name, { once = true } = {}) {
  if (!name || !config) return;
  state.current = name;
  video.classList.toggle('mirror', state.facing === 'right');
  video.loop = !once;
  video.src = '../../anim/webm/' + encodeURIComponent(name) + '.webm';
  video.play().catch(() => { /* src 变更打断上次播放会走这里，忽略 */ });
}

function scheduleNext(delayMs) {
  clearTimeout(state.restTimer);
  if (!config || state.sleeping) return;   // 睡觉时不掷动画
  const wait = delayMs != null ? delayMs
    : (config.restSeconds.min + Math.random() * (config.restSeconds.max - config.restSeconds.min)) * 1000;
  state.restTimer = setTimeout(pickNext, wait);
}

function pickCategory(categories) {
  const alive = categories.filter((cat) => usable(cat.actions).length);
  const total = alive.reduce((sum, cat) => sum + cat.weight, 0) || 1;
  let t = Math.random() * total;
  for (const cat of alive) {
    t -= cat.weight;
    if (t <= 0) return cat;
  }
  return alive[alive.length - 1];
}

function pickNext() {
  if (!config || state.dragging || state.sleeping) return;
  const w = config.weights;
  const r = Math.random() * 100;
  if (r < w.idle) return play(pick(usable(config.idle)));
  if (r < w.idle + w.turn) return play(pick(usable(config.turn)));
  if (r < w.idle + w.turn + w.move) return play(pick(usable(config.idle)));  // V1 不走路：挪到待机
  const cat = pickCategory(config.categories);
  const name = pick(usable(cat.actions));
  if (cat.noMirror && state.facing === 'right') return play(pick(usable(config.idle)));
  play(name);
}

video.addEventListener('ended', () => {
  if (state.current === '东张西望') state.facing = state.facing === 'left' ? 'right' : 'left';
  if (state.sleeping && state.current === '睡觉-开始') {
    if (usable(['睡觉-循环']).length) { play('睡觉-循环', { once: false }); return; }  // 入睡成功，循环睡
  }
  scheduleNext();          // 停留 restSeconds 再掷
});
video.addEventListener('error', () => {
  if (state.current) state.missing.add(state.current);
  scheduleNext(1500);
});

// QQ 在线联动：掉线睡着、上线醒来（由桥驱动）
function sleepNow() {
  if (!config || state.sleeping) return;
  state.sleeping = true;
  clearTimeout(state.restTimer);
  bubbleSay('大肥鱼下线了，我先睡了……', 6000);
  play('睡觉-开始');
}
function wakeUp() {
  if (!config || !state.sleeping) return;
  state.sleeping = false;
  bubbleSay('我回来啦！好耶！', 6000);
  play('睡觉-起来');
}

// ---------- 气泡 ----------
let bubbleTimer = null;
function bubbleSay(text, ms = 5200) {
  bubble.textContent = text;
  bubble.classList.add('show');
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => bubble.classList.remove('show'), ms);
}

// 大肥鱼桥：机器人侧的事件 → 气泡 + 联动动画（anim 传动画名或 'whisper' 池）
window.petAPI.onBridgeSay(({ text, anim, duration, sleep, wake }) => {
  if (wake) return wakeUp();
  if (sleep) return sleepNow();
  if (state.sleeping) return;   // 睡着时不处理日常联动
  bubbleSay(text, duration || 6000);
  if (state.dragging || !config) return;
  let name = null;
  if (anim === 'whisper') name = pick(usable(config.events && config.events.whisper));
  else if (anim) name = anim;
  if (name) {
    clearTimeout(state.restTimer);
    play(name);
  }
});

// ---------- 像素级命中：当前视频帧的 alpha ----------
let hitCtx = null;
function ensureHitCtx() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  if (!hitCtx) {
    const canvas = document.createElement('canvas');
    canvas.width = vw; canvas.height = vh;
    hitCtx = canvas.getContext('2d', { willReadFrequently: true });
  }
  return hitCtx;
}

function overWhale(clientX, clientY) {
  const rect = video.getBoundingClientRect();
  if (clientX < rect.left || clientX >= rect.right
      || clientY < rect.top || clientY >= rect.bottom) return false;
  const ctx = ensureHitCtx();
  if (!ctx) return true;   // 首帧没解码完时按窗口处理
  ctx.drawImage(video, 0, 0);
  const mirrored = video.classList.contains('mirror');
  const ix0 = Math.floor((clientX - rect.left) / rect.width * video.videoWidth);
  const iy0 = Math.floor((clientY - rect.top) / rect.height * video.videoHeight);
  // 3×3 邻域采样：边缘抗锯齿像素单点判定会随动画帧翻转，导致窗口
  // 反复切换点击穿透（拖拽抽动的根源）；任一邻点实心即算命中。
  try {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ix = Math.min(video.videoWidth - 1, Math.max(0, ix0 + dx));
        const iy = Math.min(video.videoHeight - 1, Math.max(0, iy0 + dy));
        const mx = mirrored ? video.videoWidth - 1 - ix : ix;
        if (ctx.getImageData(mx, iy, 1, 1).data[3] > 24) return true;
      }
    }
    return false;
  } catch (_) { return true; }
}

// ---------- 交互 ----------
let pressPoint = null;
let lastInteractive = null;
window.addEventListener('mousemove', (e) => {
  if (state.dragging) {
    // 拖拽期间不再评测命中：悬空动画每帧都在变 alpha，评测会让窗口
    // 在"穿透/不穿透"之间横跳，抓手被系统释放（抽动的根源）。
    if (pressPoint) window.petAPI.dragMove(e.screenX, e.screenY);
    return;
  }
  const over = overWhale(e.clientX, e.clientY);
  if (over !== lastInteractive) {
    lastInteractive = over;
    window.petAPI.setOverWhale(over);
    document.body.style.cursor = over ? 'grab' : 'default';
  }
});

window.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || state.sleeping || !overWhale(e.clientX, e.clientY)) return;
  pressPoint = { x: e.clientX, y: e.clientY, time: Date.now() };
  state.dragging = true;
  clearTimeout(state.restTimer);
  window.petAPI.setOverWhale(true);   // 拖拽期间整窗保持可交互，杜绝中途变穿透
  window.petAPI.dragStart(e.screenX - e.clientX, e.screenY - e.clientY);
  if (config) play(pick(usable(config.drag)), { once: false });   // 抓起悬空循环
});

// 松手兜底：快速甩动时光标可能甩出窗口，mouseup 收不到会卡在拖拽态。
function endDrag(resume) {
  if (!state.dragging) return;
  state.dragging = false;
  window.petAPI.dragEnd();
  window.petAPI.setOverWhale(true);
  if (resume && config) play(pick(usable(config.idle)));
}
window.addEventListener('mouseup', (e) => {
  const wasDrag = state.dragging;
  if (wasDrag) endDrag(false);
  if (pressPoint && overWhale(e.clientX, e.clientY)) {
    const moved = Math.hypot(e.clientX - pressPoint.x, e.clientY - pressPoint.y);
    if (moved < 4 && Date.now() - pressPoint.time < 500 && config) {
      pressPoint = null;
      play(pick(usable(config.clicks)));   // 点一下：点击回应动画，播完自然回流
      return;
    }
  }
  pressPoint = null;
  if (wasDrag && config) play(pick(usable(config.idle)));  // 放手回到待机链
});
window.addEventListener('blur', () => { endDrag(true); pressPoint = null; });
document.addEventListener('mouseleave', () => { if (state.dragging) { endDrag(true); pressPoint = null; } });

window.addEventListener('mouseup', (e) => {
  const wasDrag = state.dragging;
  if (wasDrag) { state.dragging = false; window.petAPI.dragEnd(); }
  if (pressPoint && overWhale(e.clientX, e.clientY)) {
    const moved = Math.hypot(e.clientX - pressPoint.x, e.clientY - pressPoint.y);
    if (moved < 4 && Date.now() - pressPoint.time < 500 && config) {
      play(pick(usable(config.clicks)));   // 点一下：点击回应动画，播完自然回流
      return;
    }
  }
  pressPoint = null;
  if (wasDrag && config) play(pick(usable(config.idle)));  // 放手回到待机链
});

window.addEventListener('contextmenu', (e) => {
  if (overWhale(e.clientX, e.clientY)) { e.preventDefault(); window.petAPI.showMenu(); }
});

window.addEventListener('wheel', (e) => {
  if (!overWhale(e.clientX, e.clientY)) return;
  e.preventDefault();
  const scale = Math.min(2.2, Math.max(0.6, (Number(config && config.sizeScale) || 1)
    * (e.deltaY < 0 ? 1.07 : 1 / 1.07)));
  if (config) config.sizeScale = scale;
  window.petAPI.resizeWindow(scale);
}, { passive: false });

// ---------- 启动 ----------
if (config) {
  // 尊重已存的体型
  (async () => {
    try {
      const saved = Number((await window.petAPI.getConfig() || {}).sizeScale) || 1;
      if (config) config.sizeScale = Math.min(2.2, Math.max(0.6, saved));
    } catch (_) {}
  })();
  // 菜单（变大/变小/恢复默认）改过体型后同步回来，滚轮不跳变
  window.petAPI.onSizeChanged((scale) => {
    if (config) config.sizeScale = Math.min(2.2, Math.max(0.6, Number(scale) || 1));
  });
  play(pick(usable(config.idle)));
  console.log('[pet] anim mode ready, idle pool:', config.idle.join(','));
}
