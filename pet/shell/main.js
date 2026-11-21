// 大肥鱼桌宠：Electron 主进程。
// 双渲染模式：webm 动画池（默认，pet/anim/webm）与 2D 精灵（退役保留）。
// 透明无边框置顶窗口；鼠标命中检测由渲染层做（当前帧像素 alpha），
// 这里只负责"整窗是否吃点击"的开关、拖拽移动、托盘藏起和右键菜单。
const { app, BrowserWindow, ipcMain, Menu, Tray, nativeImage, screen, shell } = require('electron');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
// 主进程文件日志：计划任务/隐藏启动时 stdout 不可见，崩溃遗言落这里。
const LOG_PATH = path.join(ROOT, 'pet-main.log');
function log(message) {
  try {
    fs.appendFileSync(LOG_PATH,
      `${new Date().toISOString()} ${message}\n`);
  } catch (_) { /* 日志失败不拦主流程 */ }
}
process.on('uncaughtException', (error) => {
  log(`uncaughtException: ${error && error.stack ? error.stack : error}`);
});
process.on('unhandledRejection', (reason) => {
  log(`unhandledRejection: ${reason instanceof Error ? reason.stack : reason}`);
});
const CONFIG_PATH = path.join(ROOT, 'pet-config.json');
const POOLS_PATH = path.join(ROOT, 'anim', 'pools.json');
const ANIM_DIR = path.join(ROOT, 'anim', 'webm');
const ANIM_BASE_W = 520, ANIM_BASE_H = 300;   // webm 素材是 9:16 横向比例
const SPRITE_BASE_W = 320, SPRITE_BASE_H = 430;

function loadConfig() {
  try { return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')); }
  catch (_) { return {}; }
}
function saveConfig(patch) {
  const config = { ...loadConfig(), ...patch };
  try { fs.writeFileSync(CONFIG_PATH, JSON.stringify(config)); } catch (_) {}
}
function loadPools() {
  try {
    const text = fs.readFileSync(POOLS_PATH, 'utf8')
      .split('\n')
      .filter((line) => !line.trim().startsWith('"//'))   // 去掉注释行（"//" 开头的键）
      .join('\n');
    return JSON.parse(text);
  } catch (error) {
    console.log('[pet] pools.json load failed:', String(error));
    return null;
  }
}

const animMode = fs.existsSync(POOLS_PATH)
  && fs.existsSync(path.join(ANIM_DIR, '待机呼吸休闲.webm'));
const BASE_W = animMode ? ANIM_BASE_W : SPRITE_BASE_W;
const BASE_H = animMode ? ANIM_BASE_H : SPRITE_BASE_H;

let win = null;
let tray = null;
let hiddenByTray = false;
const drag = { active: false, offsetX: 0, offsetY: 0 };

function createWindow() {
  const config = loadConfig();
  const scale = Math.min(2.2, Math.max(0.6, Number(config.sizeScale) || 1));
  win = new BrowserWindow({
    width: Math.round(BASE_W * scale),
    height: Math.round(BASE_H * scale),
    transparent: true,
    frame: false,
    hasShadow: false,
    resizable: false,
    maximizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    backgroundColor: '#00000000',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // 只加载本地文件的自足应用；关掉 webSecurity 才能用 file:// 取素材。
      webSecurity: false,
    },
  });
  win.setAlwaysOnTop(true, 'screen-saver');
  win.loadFile(path.join(__dirname, 'renderer',
    animMode ? 'anim.html' : 'sprite.html'));
  // 渲染层日志转发到 stdout，方便无 DevTools 时排障
  win.webContents.on('console-message', (_e, _level, message) => {
    console.log('[renderer]', message);
  });
  if (process.env.PET_OPAQUE) {
    win.setBackgroundColor('#1b2a3f');
  }
  // 启动位置：主屏右下角，工作区内
  const area = screen.getPrimaryDisplay().workArea;
  win.setPosition(area.x + area.width - Math.round(BASE_W * scale) - 20,
                  area.y + area.height - Math.round(BASE_H * scale) - 10);
  win.setIgnoreMouseEvents(true, { forward: true });
}

function petIcon() {
  try {
    const image = nativeImage.createFromPath(path.join(ROOT, 'sprite', 'body.png'));
    return image.isEmpty() ? null : image.resize({ width: 32, height: 32 });
  } catch (_) { return null; }
}

function createTray() {
  const image = petIcon();
  if (!image) return;   // 没图标就别建托盘，右键菜单仍可用
  tray = new Tray(image);
  tray.setToolTip('大肥鱼 · 桌宠');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: '藏起来', click: () => togglePet() },
    { type: 'separator' },
    { label: '打开日记本', click: () => shell.openExternal('http://127.0.0.1:8765/diary') },
    { label: '打开记忆星图', click: () => shell.openExternal('http://127.0.0.1:8765/memory') },
    { label: '打开 Agent 现场', click: () => shell.openExternal('http://127.0.0.1:8765/') },
    { type: 'separator' },
    { label: '大肥鱼下班（退出）', click: () => app.quit() },
  ]));
  tray.on('click', togglePet);   // 左键点托盘 = 藏/现
}

function togglePet() {
  if (!win) return;
  if (win.isVisible()) {
    hiddenByTray = true;
    win.hide();
  } else {
    hiddenByTray = false;
    win.show();
  }
  if (tray) tray.setContextMenu(Menu.buildFromTemplate(trayMenuItems()));
}

function trayMenuItems() {
  return [
    { label: hiddenByTray ? '出来' : '藏起来', click: () => togglePet() },
    { label: '变大一点', click: () => applyScale(currentScale() * 1.2) },
    { label: '变小一点', click: () => applyScale(currentScale() / 1.2) },
    { label: '恢复默认大小', click: () => applyScale(1) },
    { type: 'separator' },
    { label: '打开日记本', click: () => shell.openExternal('http://127.0.0.1:8765/diary') },
    { label: '打开记忆星图', click: () => shell.openExternal('http://127.0.0.1:8765/memory') },
    { label: '打开 Agent 现场', click: () => shell.openExternal('http://127.0.0.1:8765/') },
    { type: 'separator' },
    { label: '大肥鱼下班（退出）', click: () => app.quit() },
  ];
}

// 尺寸缩放：渲染层滚轮/菜单触发，底边中点锚定，写回配置并同步渲染层
function currentScale() {
  return Math.min(2.2, Math.max(0.6, Number(loadConfig().sizeScale) || 1));
}
function applyScale(scale) {
  if (!win) return;
  const s = Math.min(2.2, Math.max(0.6, Number(scale) || 1));
  const [x, y] = win.getPosition();
  const [oldW, oldH] = win.getSize();
  const cx = x + oldW / 2;
  const bottom = y + oldH;
  const w = Math.round(BASE_W * s);
  const h = Math.round(BASE_H * s);
  const area = screen.getDisplayNearestPoint({ x: cx, y: bottom }).workArea;
  win.setBounds({
    x: Math.max(area.x, Math.min(area.x + area.width - w, Math.round(cx - w / 2))),
    y: Math.max(area.y, Math.min(area.y + area.height - h, Math.round(bottom - h))),
    width: w, height: h,
  });
  saveConfig({ sizeScale: s });
  if (!win.isDestroyed()) win.webContents.send('size-changed', s);
}
ipcMain.on('resize-window', (_event, scale) => applyScale(scale));
ipcMain.handle('get-config', () => loadConfig());
ipcMain.handle('get-anim-config', () => loadPools());

// 命中切换：渲染层像素检测到鲸鱼时关掉整窗穿透
ipcMain.on('hit', (_event, overWhale) => {
  if (!win) return;
  try {
    win.setIgnoreMouseEvents(!overWhale, { forward: true });
  } catch (_) { /* 窗口销毁瞬间可能报错，忽略 */ }
});

// 拖拽：渲染层按住鲸鱼后把位移增量发过来
ipcMain.on('drag-start', (event, offsetX, offsetY) => {
  drag.active = true;
  drag.offsetX = offsetX;
  drag.offsetY = offsetY;
});
ipcMain.on('drag-move', (event, screenX, screenY) => {
  if (!drag.active || !win) return;
  const [x, y] = win.getPosition();
  const [w, h] = win.getSize();
  const area = screen.getDisplayNearestPoint({ x: screenX, y: screenY }).workArea;
  const nx = Math.min(Math.max(area.x, screenX - drag.offsetX), area.x + area.width - w);
  const ny = Math.min(Math.max(area.y, screenY - drag.offsetY), area.y + area.height - h);
  win.setPosition(nx, ny);
});
ipcMain.on('drag-end', () => { drag.active = false; });

// 右键菜单
ipcMain.on('ctx-menu', () => {
  if (!win) return;
  Menu.buildFromTemplate(trayMenuItems()).popup({ window: win });
});

app.whenReady().then(() => {
  log(`mode=${animMode ? 'anim' : 'sprite'} electron=${process.versions.electron}`);
  createWindow();
  createTray();
  // 大肥鱼桥：机器人侧事件 → 气泡 + 联动动画
  const { createBridge } = require('./bridge');
  const pools = loadPools() || {};
  createBridge({ win, pools, log }).start();
  log('ready: window + tray + bridge started');
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { if (tray) tray.destroy(); });
