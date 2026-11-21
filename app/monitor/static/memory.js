(() => {
  'use strict';

  const KIND = {
    taboo: { label: '禁忌', color: '#f7a6a6' },
    relationship: { label: '关系', color: '#f0a6d2' },
    profile: { label: '画像', color: '#ffd27a' },
    preference: { label: '喜好', color: '#83d7b6' },
    fact: { label: '事实', color: '#8fd4ff' },
    event: { label: '事件', color: '#b7a8ff' },
    'running_joke': { label: '梗', color: '#ffc98a' },
    decision: { label: '决定', color: '#9be8e0' },
    plan: { label: '计划', color: '#c9d68a' },
    current: { label: '近况', color: '#9b9da8' },
  };
  const KIND_FALLBACK = { label: '其它', color: '#8a90a3' };
  const ARCHIVE_LIVES = 6.0;

  const canvas = document.getElementById('sky');
  const ctx = canvas.getContext('2d');
  const tip = document.getElementById('tip');
  const statusEl = document.getElementById('status');
  const generatedEl = document.getElementById('generated');
  const statsEl = document.getElementById('stats');
  const legendEl = document.getElementById('legend');
  const fadingEl = document.getElementById('fading');
  const emptyEl = document.getElementById('empty');
  const emptyReason = document.getElementById('empty-reason');

  let state = { available: false, nodes: [], stats: {}, generated_at: '', reason: '' };
  let layout = [];
  let screen = [];
  let hovered = null;
  let pinned = null;
  let view = { zoom: 1, tx: 0, ty: 0 };
  let dragging = false;
  let moved = false;
  let last = { x: 0, y: 0 };
  let dpr = window.devicePixelRatio || 1;
  let bg = null;

  const hash = (value) => {
    let h = 2166136261;
    const text = String(value);
    for (let i = 0; i < text.length; i += 1) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) / 4294967295;
  };

  const kindOf = (kind) => KIND[kind] || KIND_FALLBACK;

  const fmtDays = (days) => {
    if (days === null || days === undefined) return '永不遗忘';
    if (days <= 0) return '已到线';
    if (days < 1) return '不到1天';
    if (days < 10) return days.toFixed(1) + ' 天';
    return Math.round(days) + ' 天';
  };

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    bg = buildBackground(canvas.width, canvas.height);
  }

  function buildBackground(width, height) {
    const layer = document.createElement('canvas');
    layer.width = width;
    layer.height = height;
    const g = layer.getContext('2d');
    const count = Math.round((width * height) / 9000);
    for (let i = 0; i < count; i += 1) {
      const x = hash('bx' + i) * width;
      const y = hash('by' + i) * height;
      const r = (0.3 + hash('br' + i) * 1.3) * dpr;
      const a = 0.05 + hash('ba' + i) * 0.22;
      g.fillStyle = 'rgba(200,214,255,' + a.toFixed(3) + ')';
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }
    return layer;
  }

  function computeLayout() {
    const nodes = state.nodes || [];
    const n = nodes.length;
    const golden = 2.399963229728653;
    layout = nodes.map((node, index) => {
      // 强度高的靠近核，弱的散在外圈；哈希抖动让排布像星云而不是等距螺旋。
      const t = n <= 1 ? 0 : index / (n - 1);
      const radius = 0.16 + 0.82 * Math.sqrt(t);
      const jitter = (hash('j' + node.id) - 0.5) * 0.1;
      return {
        node,
        angle: index * golden + (hash('a' + node.id) - 0.5) * 0.24,
        radius: Math.max(0.1, radius + jitter),
        phase: hash('p' + node.id) * Math.PI * 2,
      };
    });
    screen = new Array(layout.length);
  }

  async function load(force) {
    statusEl.textContent = '正在读取快照';
    statusEl.className = 'chip';
    try {
      const response = await fetch('/api/memory-map', { cache: 'no-store' });
      if (!response.ok) throw new Error('http ' + response.status);
      const data = await response.json();
      state = {
        available: Boolean(data.available),
        nodes: Array.isArray(data.nodes) ? data.nodes : [],
        stats: data.stats && typeof data.stats === 'object' ? data.stats : {},
        generated_at: data.generated_at || '',
        reason: data.reason || '',
      };
      computeLayout();
      renderPanels();
      paintStatus();
      if (force) pinned = null;
    } catch (error) {
      statusEl.textContent = '读取失败';
      statusEl.className = 'chip error';
      generatedEl.textContent = String(error && error.message ? error.message : error);
    }
  }

  function paintStatus() {
    const empty = !state.available || state.nodes.length === 0;
    emptyEl.hidden = !empty;
    if (empty) {
      const reasons = {
        snapshot_missing: '记忆快照还没生成：等机器人跑一次快照（最长 5 分钟），再点刷新。',
        snapshot_unreadable: '快照文件读不出来，等下一轮覆盖。',
        snapshot_too_large: '快照文件异常偏大，已拒绝加载。',
        snapshot_incompatible: '快照版本不匹配，等机器人升级后重写。',
      };
      emptyReason.textContent = state.available
        ? '目前没有活跃记忆。'
        : (reasons[state.reason] || '记忆快照暂不可用。');
    }
    if (state.available) {
      statusEl.textContent = '星图在线';
      statusEl.className = 'chip ok';
      generatedEl.textContent = '快照 ' + (state.generated_at || '未知时间');
    } else {
      statusEl.textContent = '等待快照';
      statusEl.className = 'chip error';
      generatedEl.textContent = '';
    }
  }

  function renderPanels() {
    const stats = state.stats || {};
    const items = [
      ['活跃记忆', stats.total || 0],
      ['永不遗忘', stats.immutable || 0],
      ['身份级', stats.promoted || 0],
      ['淡忘中', stats.faded || 0],
      ['7天内归档', stats.archiving_soon || 0],
      ['记忆类型', Object.keys(stats.by_kind || {}).length],
    ];
    statsEl.replaceChildren(...items.map(([label, value]) => {
      const box = document.createElement('div');
      box.className = 'stat';
      const span = document.createElement('span');
      span.textContent = label;
      const strong = document.createElement('strong');
      strong.textContent = String(value);
      box.append(span, strong);
      return box;
    }));

    const byKind = stats.by_kind || {};
    const rows = Object.entries(byKind).sort((a, b) => b[1] - a[1]).map(([kind, count]) => {
      const info = kindOf(kind);
      const row = document.createElement('div');
      row.className = 'legend-row';
      const dot = document.createElement('i');
      dot.style.background = info.color;
      const label = document.createElement('span');
      label.textContent = info.label + ' · ' + kind;
      const total = document.createElement('b');
      total.textContent = String(count);
      row.append(dot, label, total);
      return row;
    });
    legendEl.replaceChildren(...rows);

    const fading = state.nodes
      .filter((node) => !node.immutable && node.archive_in_days !== null)
      .sort((a, b) => a.archive_in_days - b.archive_in_days)
      .slice(0, 8);
    if (!fading.length) {
      const p = document.createElement('p');
      p.className = 'muted small';
      p.textContent = '没有临近归档的记忆。';
      fadingEl.replaceChildren(p);
      return;
    }
    fadingEl.replaceChildren(...fading.map((node) => {
      const button = document.createElement('button');
      button.type = 'button';
      const text = document.createElement('span');
      text.className = 'text';
      text.textContent = node.content || '（无内容）';
      const row = document.createElement('span');
      row.className = 'row';
      const kind = document.createElement('span');
      kind.textContent = kindOf(node.kind).label;
      const left = document.createElement('span');
      left.className = node.archive_in_days <= 3 ? 'urgent' : '';
      left.textContent = '还剩 ' + fmtDays(node.archive_in_days) + '归档';
      row.append(kind, left);
      button.append(text, row);
      button.addEventListener('click', () => focusNode(node.id));
      return button;
    }));
  }

  function focusNode(id) {
    pinned = id;
    hovered = null;
    const index = layout.findIndex((item) => item.node.id === id);
    if (index < 0) return;
    const item = layout[index];
    const size = Math.min(canvas.width, canvas.height);
    const worldR = item.radius * size * 0.46 * view.zoom;
    const px = Math.cos(item.angle) * worldR;
    const py = Math.sin(item.angle) * worldR;
    view.tx = -px;
    view.ty = -py;
  }

  function draw(nowMs) {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (bg) ctx.drawImage(bg, 0, 0);

    const cx = width / 2 + view.tx * dpr;
    const cy = height / 2 + view.ty * dpr;
    const size = Math.min(width, height);
    const maxR = size * 0.46 * view.zoom;
    const time = nowMs / 1000;

    // 星图核心
    const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 0.16);
    coreGlow.addColorStop(0, 'rgba(143,184,255,.34)');
    coreGlow.addColorStop(0.45, 'rgba(110,140,220,.12)');
    coreGlow.addColorStop(1, 'rgba(110,140,220,0)');
    ctx.fillStyle = coreGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.16, 0, Math.PI * 2);
    ctx.fill();
    const pulse = 1 + Math.sin(time * 0.9) * 0.06;
    ctx.fillStyle = '#dce8ff';
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.014 * pulse, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = '600 ' + Math.round(12 * dpr) + 'px "Segoe UI","Microsoft YaHei",sans-serif';
    ctx.fillStyle = 'rgba(220,232,255,.82)';
    ctx.textAlign = 'center';
    ctx.fillText('大肥鱼', cx, cy + size * 0.045);

    screen = new Array(layout.length);
    for (let i = 0; i < layout.length; i += 1) {
      const item = layout[i];
      const node = item.node;
      const r = item.radius * maxR;
      const x = cx + Math.cos(item.angle) * r;
      const y = cy + Math.sin(item.angle) * r;
      const strength = Math.max(0, Math.min(1, Number(node.strength) || 0));
      const twinkle = 0.86 + 0.14 * Math.sin(time * 1.15 + item.phase);
      const active = node.id === hovered || node.id === pinned;
      const radius = (2.2 + 5.2 * strength) * dpr * (active ? 1.55 : 1);
      const info = kindOf(node.kind);
      const alpha = (0.22 + 0.78 * strength) * twinkle;

      ctx.strokeStyle = 'rgba(143,184,255,' + (0.03 + 0.1 * strength).toFixed(3) + ')';
      ctx.lineWidth = Math.max(0.5, dpr * 0.6);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();

      if (node.immutable) {
        ctx.strokeStyle = 'rgba(255,210,122,' + (0.5 + 0.3 * twinkle).toFixed(3) + ')';
        ctx.lineWidth = Math.max(1, dpr * 1.1);
        ctx.beginPath();
        ctx.arc(x, y, radius + 3.5 * dpr, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (node.promoted && !node.immutable) {
        ctx.strokeStyle = 'rgba(155,232,224,.55)';
        ctx.lineWidth = Math.max(0.8, dpr);
        ctx.beginPath();
        ctx.arc(x, y, radius + 3 * dpr, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (node.faded) {
        ctx.setLineDash([2 * dpr, 2 * dpr]);
        ctx.strokeStyle = 'rgba(155,157,168,.55)';
        ctx.lineWidth = Math.max(0.8, dpr);
        ctx.beginPath();
        ctx.arc(x, y, radius + 5.5 * dpr, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const color = node.faded ? 'rgba(150,153,168,' + alpha.toFixed(3) + ')' : hexToRgba(info.color, alpha);
      const glow = ctx.createRadialGradient(x, y, 0, x, y, radius * 3.2);
      glow.addColorStop(0, hexToRgba(info.color, Math.min(0.55, alpha * 0.55)));
      glow.addColorStop(1, hexToRgba(info.color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, radius * 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      screen[i] = { x: x / dpr, y: y / dpr, radius: radius / dpr, node };
    }

    const focus = pinned !== null ? pinned : hovered;
    if (focus !== null) {
      const entry = screen.find((item) => item && item.node.id === focus);
      if (entry) {
        ctx.font = '600 ' + Math.round(12 * dpr) + 'px "Segoe UI","Microsoft YaHei",sans-serif';
        ctx.fillStyle = '#ececef';
        ctx.textAlign = 'left';
        const label = String(entry.node.content || '').slice(0, 18);
        ctx.fillText(label, entry.x * dpr + entry.radius * dpr + 6 * dpr, entry.y * dpr - 4 * dpr);
      }
    }
  }

  function hexToRgba(hex, alpha) {
    const value = hex.replace('#', '');
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + Math.max(0, Math.min(1, alpha)).toFixed(3) + ')';
  }

  function loop(ms) {
    draw(ms);
    requestAnimationFrame(loop);
  }

  function nearest(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best = null;
    let bestDistance = 16;
    for (const entry of screen) {
      if (!entry) continue;
      const dx = entry.x - x;
      const dy = entry.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < Math.max(bestDistance, entry.radius + 6)) {
        bestDistance = distance;
        best = entry;
      }
    }
    return best;
  }

  function showTip(entry) {
    if (!entry) {
      tip.hidden = true;
      return;
    }
    const node = entry.node;
    const info = kindOf(node.kind);
    tip.replaceChildren();
    const title = document.createElement('strong');
    title.textContent = node.content || '（无内容）';
    const meta = document.createElement('div');
    meta.className = 'meta';
    const lines = [
      info.label + ' · ' + node.kind + (node.immutable ? ' · 永不遗忘' : node.promoted ? ' · 身份级' : ''),
      '强度 ' + Math.round((Number(node.strength) || 0) * 100) + '%',
      '半衰期 ' + node.half_life_days + ' 天 · 提及 ' + node.mention_count + ' 次',
      '最近提起 ' + (node.last_seen_at || '未知'),
      '归档倒计时 ' + fmtDays(node.archive_in_days),
    ];
    for (const line of lines) {
      const div = document.createElement('div');
      div.textContent = line;
      meta.append(div);
    }
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('i');
    fill.style.width = Math.round(Math.max(0, Math.min(1, Number(node.strength) || 0)) * 100) + '%';
    bar.append(fill);
    tip.append(title, bar, meta);
    const rect = canvas.getBoundingClientRect();
    const left = Math.min(rect.width - 372, entry.x + 16);
    const top = Math.max(8, Math.min(rect.height - 150, entry.y - 16));
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = top + 'px';
    tip.hidden = false;
  }

  canvas.addEventListener('mousemove', (event) => {
    if (dragging) {
      const dx = event.clientX - last.x;
      const dy = event.clientY - last.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      view.tx += dx;
      view.ty += dy;
      last = { x: event.clientX, y: event.clientY };
      return;
    }
    const entry = nearest(event.clientX, event.clientY);
    hovered = entry ? entry.node.id : null;
    showTip(entry);
  });

  canvas.addEventListener('mousedown', (event) => {
    dragging = true;
    moved = false;
    last = { x: event.clientX, y: event.clientY };
    canvas.classList.add('dragging');
  });

  window.addEventListener('mouseup', (event) => {
    const wasDragging = dragging && moved;
    dragging = false;
    canvas.classList.remove('dragging');
    if (wasDragging) return;
    const entry = nearest(event.clientX, event.clientY);
    if (entry && canvas.contains(event.target)) {
      pinned = pinned === entry.node.id ? null : entry.node.id;
      showTip(entry);
    } else if (canvas.contains(event.target)) {
      pinned = null;
    }
  });

  canvas.addEventListener('mouseleave', () => {
    hovered = null;
    if (pinned === null) tip.hidden = true;
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    view.zoom = Math.max(0.55, Math.min(3.2, view.zoom * factor));
  }, { passive: false });

  document.getElementById('refresh').addEventListener('click', () => { load(true); });

  window.addEventListener('resize', resize);

  resize();
  load(false);
  requestAnimationFrame(loop);
  window.setInterval(() => { load(false); }, 60000);
})();
