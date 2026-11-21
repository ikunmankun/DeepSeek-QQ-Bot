// 渲染层与主进程之间唯一通道：命中切换、拖拽、菜单、动画配置、桥事件。
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('petAPI', {
  setOverWhale: (over) => ipcRenderer.send('hit', Boolean(over)),
  dragStart: (offsetX, offsetY) => ipcRenderer.send('drag-start', offsetX, offsetY),
  dragMove: (screenX, screenY) => ipcRenderer.send('drag-move', screenX, screenY),
  dragEnd: () => ipcRenderer.send('drag-end'),
  showMenu: () => ipcRenderer.send('ctx-menu'),
  resizeWindow: (scale) => ipcRenderer.send('resize-window', scale),
  getConfig: () => ipcRenderer.invoke('get-config'),
  getAnimConfig: () => ipcRenderer.invoke('get-anim-config'),
  onResetSize: (callback) => ipcRenderer.on('reset-size', () => callback()),
  onSizeChanged: (callback) => ipcRenderer.on('size-changed', (_event, scale) => callback(scale)),
  onBridgeSay: (callback) => ipcRenderer.on('bridge-say', (_event, payload) => callback(payload)),
});
