// 大肥鱼桌宠渲染层：three.js 加载 GLB + 全程序化动画（无骨骼依赖）。
// 动作：待机漂浮/呼吸、被拖拽游动、摸头挤压回弹、双击转圈、深夜犯困。
// 状态接真实数据：夜间按 bot 的安静时段睡觉，气泡偶尔报记忆星图数字。
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const stage = document.getElementById('stage');
const bubble = document.getElementById('bubble');

// ---------- 场景 ----------
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(32, window.innerWidth / window.innerHeight, 0.1, 50);
camera.position.set(0, 0.25, 3.4);
camera.lookAt(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 1.05));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(1.6, 2.2, 2.4);
scene.add(key);
const rim = new THREE.DirectionalLight(0x9db8ff, 0.7);
rim.position.set(-2, 1, -1.6);
scene.add(rim);

// ---------- 模型 ----------
const whale = new THREE.Group();
scene.add(whale);
let baseScale = 1;
let mixer = null;            // 骨骼动画（Tripo 绑定导出的 GLB 自带 clips）
const actions = {};
let currentAction = null;
function playNamed(pattern, once = true) {
  if (!mixer) return false;
  const hit = Object.keys(actions).find((name) => pattern.test(name));
  if (!hit) return false;
  const next = actions[hit];
  if (currentAction === next) return true;
  next.reset();
  if (currentAction) currentAction.fadeOut(0.15);
  next.fadeIn(0.15).play();
  if (once) {
    const idleHit = Object.keys(actions).find((name) => /idle|待机/i.test(name));
    if (idleHit && actions[idleHit] !== next) {
      const idle = actions[idleHit];
      const duration = (next.getClip().duration || 1) / Math.max(0.01, next.timeScale);
      setTimeout(() => { idle.reset(); next.fadeOut(0.15); idle.fadeIn(0.15).play();
        if (currentAction === next) currentAction = idle; }, duration * 1000);
    }
  }
  currentAction = next;
  return true;
}
// 从顶点 X 直方图找空隙，切出最左边的角色团（三视图拼图被合并生成的场合）。
// 找不到空隙（真正的单角色模型）就退回左 1/3，同样成立。
function pickFrontBand(model, box, size) {
  const fallback = () => new THREE.Box3(
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x + size.x / 3, box.max.y, box.max.z));
  let positions = null;
  model.traverse((obj) => {
    if (positions || !obj.isMesh) return;
    const attr = obj.geometry && obj.geometry.getAttribute('position');
    if (attr) positions = attr.array;
  });
  if (!positions || size.x <= 1e-6) return fallback();
  const bins = 96;
  const hist = new Array(bins).fill(0);
  for (let i = 0; i < positions.length; i += 3) {
    const b = Math.floor((positions[i] - box.min.x) / size.x * bins);
    hist[Math.min(bins - 1, Math.max(0, b))] += 1;
  }
  const gaps = [];
  let run = 0;
  for (let b = 0; b < bins; b += 1) {
    if (hist[b] === 0) run += 1;
    else { if (run >= 3) gaps.push(b); run = 0; }
  }
  if (!gaps.length) return fallback();
  const split = box.min.x + (gaps[0] / bins) * size.x;
  return new THREE.Box3(new THREE.Vector3(box.min.x, box.min.y, box.min.z),
                        new THREE.Vector3(split, box.max.y, box.max.z));
}

console.log('[pet] loading model');
new GLTFLoader().load('../../dafeiyu.glb', (gltf) => {
  console.log('[pet] model loaded');
  const model = gltf.scene;
  // 上游坑：三视图拼图被 Tripo 当单图生成，网格里是三个角色并排。
  // 自动找空隙取最左团（正面角色）作为主角居中。
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const bandBox = pickFrontBand(model, box, size);
  const center3 = new THREE.Vector3();
  bandBox.getCenter(center3);
  const bandSize = bandBox.getSize(new THREE.Vector3());
  baseScale = (Math.max(bandSize.x, bandSize.y, bandSize.z) || 1) * 0.58;
  model.scale.setScalar(1 / baseScale);
  model.position.sub(center3.divideScalar(baseScale));
  whale.add(model);
  whale.visible = true;
  // Tripo 绑定导出的 GLB 自带动画 clips：有就用真的，并默认循环 idle
  if (gltf.animations && gltf.animations.length) {
    mixer = new THREE.AnimationMixer(model);
    for (const clip of gltf.animations) actions[clip.name.toLowerCase()] = mixer.clipAction(clip);
    const idleClip = gltf.animations.find((c) => /idle|待机/i.test(c.name)) || gltf.animations[0];
    actions[idleClip.name.toLowerCase()].play();
    currentAction = actions[idleClip.name.toLowerCase()];
    console.log('[pet] skeletal animations:', gltf.animations.map((c) => c.name).join(', '));
  }
  console.log('[pet] whale on stage, scale=', baseScale.toFixed(3),
              'band=', bandSize.x.toFixed(2), 'of', size.x.toFixed(2));
}, undefined, (error) => {
  console.log('[pet] model load failed:', error);
  bubbleSay('模型加载失败了');
});

// ---------- 台词 ----------
const IDLE_LINES = [
  '干啥呢，饭点都不叫我',
  '这活费token，加饭',
  '别戳了，气鼓鱼也是鱼',
  '今天的记忆星图亮不亮？',
  '夜宵考虑一下白饭？',
  '在群里说话要像随手打字，我练过了',
  '谁又发表情包了，我看看',
  '摸鱼可以，摸我请排队',
  '睡什么睡，起来嗨……算了，困',
];
const PAT_LINES = [
  '哼，手挺凉',
  '再戳就鼓成球了',
  '行吧，准你摸这一下',
  '鱼离不开水，我离不开饭',
  '哦。',
];
const HAPPY_LINES = [
  '好耶！',
  'token 到账，开炫',
  '今天也是被需要的一天',
];

let bubbleTimer = null;
function bubbleSay(text, ms = 4200) {
  bubble.textContent = text;
  bubble.classList.add('show');
  state.hopT = 0;                       // 说话配一个小跳
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => bubble.classList.remove('show'), ms);
}
const pick = (list) => list[Math.floor(Math.random() * list.length)];

// ---------- 程序化动画状态 ----------
const state = {
  mode: 'idle',            // idle | sleep
  patT: -1,                // 摸头回弹的相位（秒），-1 = 不在动作中
  spinT: -1,               // 双击转圈相位
  hopT: -1,                // 冒泡小跳相位
  danceT: -1,              // 随机跳舞相位
  danceNext: 40 + Math.random() * 80,   // 距下次跳舞的秒数
  bobPhase: Math.random() * 6,
  dragging: false,
  lookX: 0, lookTarget: 0, // 视线跟随鼠标
};

function isNight(now = new Date()) {
  const h = now.getHours();
  return h >= 23 || h < 8;  // 与 bot 的安静时段一致
}

// ---------- 交互：命中检测 / 拖拽 / 点击 / 右键 ----------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
function overWhale(clientX, clientY) {
  if (!whale.children.length) return false;
  pointer.x = (clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObject(whale, true).length > 0;
}

let pressPoint = null;
let lastInteractive = null;
window.addEventListener('mousemove', (e) => {
  const over = overWhale(e.clientX, e.clientY);
  if (over !== lastInteractive) {
    lastInteractive = over;
    window.petAPI.setOverWhale(over);
    stage.style.cursor = over ? 'grab' : 'default';
  }
  state.lookTarget = ((e.clientX / window.innerWidth) - 0.5) * 0.55;
  if (state.dragging && pressPoint) {
    window.petAPI.dragMove(e.screenX, e.screenY);
  }
});

// 滚轮缩放：悬停在鲸鱼上滚动即可放大/缩小，尺寸写回配置
window.addEventListener('wheel', (e) => {
  if (!overWhale(e.clientX, e.clientY)) return;
  e.preventDefault();
  sizeScale = Math.min(2.2, Math.max(0.6, sizeScale * (e.deltaY < 0 ? 1.07 : 1 / 1.07)));
  window.petAPI.resizeWindow(sizeScale);
}, { passive: false });

window.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || !overWhale(e.clientX, e.clientY)) return;
  pressPoint = { x: e.clientX, y: e.clientY, time: Date.now() };
  state.dragging = true;
  // 窗口原点在屏幕上的位置 = 鼠标屏幕坐标 - 鼠标窗口内坐标
  window.petAPI.dragStart(e.screenX - e.clientX, e.screenY - e.clientY);
});

window.addEventListener('mouseup', (e) => {
  if (state.dragging) {
    state.dragging = false;
    window.petAPI.dragEnd();
  }
  if (pressPoint && overWhale(e.clientX, e.clientY)) {
    const moved = Math.hypot(e.clientX - pressPoint.x, e.clientY - pressPoint.y);
    if (moved < 4 && Date.now() - pressPoint.time < 500) {
      state.patT = 0;                      // 摸头：挤压回弹
      if (!playNamed(/pat|wave|摸/i)) bubbleSay(pick(PAT_LINES));
      else bubbleSay(pick(PAT_LINES));
    }
  }
  pressPoint = null;
});

window.addEventListener('dblclick', (e) => {
  if (overWhale(e.clientX, e.clientY)) {
    state.spinT = 0;                       // 双击：开心转圈
    playNamed(/happy|dance|jump|开心|舞/i, true);
    bubbleSay(pick(HAPPY_LINES));
  }
});

window.addEventListener('contextmenu', (e) => {
  if (overWhale(e.clientX, e.clientY)) {
    e.preventDefault();
    window.petAPI.showMenu();
  }
});

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

// ---------- 启动：读取尺寸配置 ----------
let sizeScale = 1;
(async () => {
  try {
    const config = await window.petAPI.getConfig();
    const saved = Math.min(2.2, Math.max(0.6, Number(config && config.sizeScale) || 1));
    if (Math.abs(saved - 1) > 0.01) {
      sizeScale = saved;
      window.petAPI.resizeWindow(sizeScale);
    }
  } catch (_) { /* 配置读不到就用默认 */ }
})();
window.petAPI.onResetSize(() => {
  sizeScale = 1;
  window.petAPI.resizeWindow(1);
  bubbleSay('恢复默认体型');
});

// ---------- 主动冒泡：闲聊 + 记忆星图播报 ----------
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
  } catch (_) { /* bot 没在线就安静待着 */ }
}
setInterval(memoryReport, 10 * 60 * 1000);
setTimeout(memoryReport, 8000);

// ---------- 主循环 ----------
const clock = new THREE.Clock();
let firstFrameLogged = false;
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;
  if (mixer) mixer.update(dt);
  if (!firstFrameLogged) {
    firstFrameLogged = true;
    console.log('[pet] first frame rendered, webgl=', renderer.getContext() ? 'ok' : 'none');
  }
  const night = isNight();

  // 待机：漂浮 + 呼吸；夜间变慢变矮
  const bobSpeed = night ? 0.5 : 1.25;
  const bobAmp = night ? 0.015 : 0.035;
  const breathe = 1 + Math.sin(t * bobSpeed + state.bobPhase) * bobAmp;

  let squashY = 1;
  if (state.patT >= 0) {
    state.patT += dt;
    // 挤压回弹：衰减正弦，0.9s 结束
    const decay = Math.exp(-state.patT * 5.2);
    squashY = 1 - 0.16 * Math.sin(state.patT * 17) * decay;
    if (state.patT > 0.9) state.patT = -1;
  }

  // 模型已在加载时按 baseScale 归一，这里只叠呼吸/挤压，避免双重缩放
  whale.scale.setScalar(breathe);
  whale.scale.y *= squashY;
  whale.scale.x *= 2 - squashY;
  whale.scale.z *= 2 - squashY;

  whale.position.y = Math.sin(t * bobSpeed + state.bobPhase) * bobAmp;

  // 视线跟随鼠标（转圈动作期间让位）
  state.lookX += (state.lookTarget - state.lookX) * Math.min(1, dt * 6);

  // 冒泡小跳
  if (state.hopT >= 0) {
    state.hopT += dt;
    const p = Math.min(1, state.hopT / 0.5);
    whale.position.y += Math.sin(p * Math.PI) * 0.07;
    if (p >= 1) state.hopT = -1;
  }

  // 偶尔来一段左右摇摆的小舞蹈
  state.danceNext -= dt;
  if (state.danceNext <= 0 && !night && !state.dragging) {
    state.danceT = 0;
    state.danceNext = 120 + Math.random() * 180;
  }
  if (state.danceT >= 0) {
    state.danceT += dt;
    const p = state.danceT / 2.4;
    whale.rotation.z += Math.sin(state.danceT * 10) * 0.14;
    whale.position.y += Math.abs(Math.sin(state.danceT * 8)) * 0.05;
    if (p >= 1) state.danceT = -1;
  }

  if (state.spinT >= 0) {
    state.spinT += dt;
    const p = Math.min(1, state.spinT / 1.1);
    whale.rotation.y = state.lookX * (1 - p) + p * Math.PI * 4 * (1 - Math.pow(p, 2) * 0.15);
    whale.scale.multiplyScalar(1 + 0.12 * Math.sin(p * Math.PI));
    if (state.spinT > 1.1) { state.spinT = -1; whale.rotation.y = 0; }
  } else {
    whale.rotation.y = state.lookX + Math.sin(t * 0.35) * 0.08;
  }

  renderer.render(scene, camera);
}
animate();
