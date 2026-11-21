// 大肥鱼桥（桌宠壳内置版）：把机器人和 QQ 的真实状态喂给桌宠。
// 每 20 秒轮询监控 API（pet-events / memory-map / diary）做日常联动；
// 每 45 秒直连 OneBot WebSocket 发 get_status 探测 QQ 在线态——
// 掉线（连续两次）桌宠睡觉，恢复上线叫醒。逻辑移植自退役的
// pet/bridge/pet_bridge.py watch()，在线探测为新增。
// 待办同步（dsh-pet 的 todos 窗口）V1 未做：自研壳暂无待办 UI。
const WebSocket = require('ws');

const MONITOR = 'http://127.0.0.1:8765';
const ONEBOT_WS = 'ws://127.0.0.1:3001';
const POLL_SECONDS = 20;
const QQ_PROBE_SECONDS = 45;

const OWNER_ARRIVE_ANIMS = ['点击回应-开心跃动', '点击回应-元气挥手', '点击回应-害羞惊讶'];

function createBridge({ win, pools, log = () => {} }) {
  const state = {
    seenMessageIds: new Set(),
    seenWishIds: new Set(),
    seenGroupIds: new Set(),
    seenChatIds: new Set(),
    firstEventsTick: true,
    announcedDiaryDates: new Set(),
    lastWhisper: 0,
    lastGroupReact: 0,
    lastChatReact: 0,
    lastMemoryTick: 0,
    lastMemoryKey: null,
    qqOnline: null,          // null=尚未探测成功
    qqOfflineStrikes: 0,
    timer: null,
    qqTimer: null,
    stopped: false,
  };

  function say(payload) {
    if (!win || win.isDestroyed()) return;
    const out = { ...payload };
    if (out.anim === 'whisper' && pools.events && pools.events.whisper) {
      const pool = pools.events.whisper;
      out.anim = pool[Math.floor(Math.random() * pool.length)];
    }
    win.webContents.send('bridge-say', out);
  }

  async function getJson(path, timeoutMs = 8000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(MONITOR + path, { cache: 'no-store', signal: controller.signal });
      return await resp.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // ---- QQ 在线探测：OneBot ws 短连接发 get_status，读 data.online ----
  function probeQQ() {
    return new Promise((resolve) => {
      let settled = false;
      const finish = (online) => {
        if (settled) return;
        settled = true;
        try { socket.close(); } catch (_) {}
        resolve(online);
      };
      let socket;
      try {
        socket = new WebSocket(ONEBOT_WS);
      } catch (_) { return finish(false); }
      const timeout = setTimeout(() => finish(false), 5000);
      socket.on('open', () => {
        socket.send(JSON.stringify({ action: 'get_status', echo: 'pet-probe' }));
      });
      socket.on('message', (buffer) => {
        try {
          const reply = JSON.parse(buffer.toString());
          // 连接建立后 napcat 会先推 lifecycle 等事件，只认带 echo 的回执
          if (reply.echo !== 'pet-probe') return;
          const online = Boolean(reply.data && reply.data.online);
          clearTimeout(timeout);
          finish(online);
        } catch (_) { /* 事件推送解析失败忽略 */ }
      });
      socket.on('error', () => { clearTimeout(timeout); finish(false); });
      socket.on('close', () => { clearTimeout(timeout); finish(false); });
    });
  }

  async function qqTick() {
    if (state.stopped) return;
    const online = await probeQQ();
    log(`bridge: qq probe online=${online}`);
    if (online) {
      state.qqOfflineStrikes = 0;
      if (state.qqOnline === false) {
        say({ wake: true });          // 醒来 + "我回来啦"（渲染层说）
        log('bridge: qq back online -> wake');
      }
      state.qqOnline = true;
      return;
    }
    state.qqOfflineStrikes += 1;
    // 连续两次探测失败才算掉线，避免网络抖动误判
    if (state.qqOnline !== false && state.qqOfflineStrikes >= 2) {
      state.qqOnline = false;
      say({ sleep: true });
      log('bridge: qq offline x2 -> sleep');
    }
  }

  // ---- 日常联动轮询 ----
  async function tick() {
    if (state.stopped) return;
    const now = Date.now();
    try {
      const events = await getJson('/api/pet-events');
      if (events && events.available) {
        // 主人私聊新消息：挥手/跃动表看到（限流 3 分钟），显示更完整内容
        const fresh = (events.owner_messages || [])
          .filter((m) => !state.seenMessageIds.has(m.id));
        for (const m of events.owner_messages || []) state.seenMessageIds.add(m.id);
        if (fresh.length && now - state.lastWhisper > 180000) {
          const last = fresh[fresh.length - 1];
          say({ text: '主人在私聊叫我了：' + String(last.text || '').slice(0, 120),
            anim: OWNER_ARRIVE_ANIMS[Math.floor(Math.random() * OWNER_ARRIVE_ANIMS.length)] });
          state.lastWhisper = now;
        }
        // 愿望实现：女仆舞庆祝
        for (const wish of events.wishes_done || []) {
          if (state.seenWishIds.has(wish.id)) continue;
          state.seenWishIds.add(wish.id);
          say({ text: '愿望实现了！' + String(wish.content || '').slice(0, 60),
            anim: '优雅女仆舞', duration: 8000 });
        }
        // 群聊动静（bot 自己回复或被 @；限流 8 分钟；首轮只记录不反应）
        const groupFresh = [];
        for (const g of events.group_activity || []) {
          if (state.seenGroupIds.has(g.id)) continue;
          state.seenGroupIds.add(g.id);
          if (!state.firstEventsTick) groupFresh.push(g);
        }
        if (groupFresh.length && now - state.lastGroupReact > 480000) {
          const last = groupFresh[groupFresh.length - 1];
          say({ text: (last.by_bot ? '刚在群里回了句：' : '群里有人叫我：')
            + String(last.text || '').slice(0, 50),
            anim: last.by_bot ? '原地漂浮踏步' : '东张西望' });
          state.lastGroupReact = now;
        }
        // 群消息播报（任何群友的发言，带群名片/昵称；限流 5 分钟）
        const chatFresh = [];
        for (const c of events.group_chat || []) {
          if (state.seenChatIds.has(c.id)) continue;
          state.seenChatIds.add(c.id);
          if (!state.firstEventsTick) chatFresh.push(c);
        }
        if (chatFresh.length && now - state.lastChatReact > 300000) {
          const last = chatFresh[chatFresh.length - 1];
          say({ text: `群里${last.user || '有人说'}：` + String(last.text || '').slice(0, 60),
            anim: '东张西望' });
          state.lastChatReact = now;
        }
      }
    } catch (_) { /* 监控不在线就静默，下轮再试 */ }
    state.firstEventsTick = false;

    // 记忆星图播报（低频，600 秒）
    if (now - state.lastMemoryTick > 600000) {
      state.lastMemoryTick = now;
      try {
        const map = await getJson('/api/memory-map');
        if (map && map.available && map.stats && map.stats.total) {
          const key = String(map.stats.total) + '/' + String(map.stats.immutable);
          if (key !== state.lastMemoryKey) {
            say({ text: `现在记得 ${map.stats.total} 件事，${map.stats.immutable} 件永不遗忘，`
              + `${map.stats.faded} 件在淡忘` });
            state.lastMemoryKey = key;
          }
        }
      } catch (_) {}
    }

    // 日记写好提示（每晚一次）
    try {
      const diary = await getJson('/api/diary');
      if (diary && diary.available && Array.isArray(diary.days) && diary.days.length) {
        const latest = diary.days[0];
        if (latest.reflected && !state.announcedDiaryDates.has(latest.date)) {
          state.announcedDiaryDates.add(latest.date);
          const reflection = (latest.entries || []).find((entry) => entry.reflection);
          say({ text: '今晚的日记写好了：' + (reflection ? reflection.text.slice(0, 40) + '…' : latest.date)
            + '\n（网页端 /diary 可看）', anim: 'whisper', duration: 8000 });
        }
      }
    } catch (_) {}
  }

  return {
    start() {
      if (state.timer) return;
      state.timer = setInterval(tick, POLL_SECONDS * 1000);
      state.qqTimer = setInterval(qqTick, QQ_PROBE_SECONDS * 1000);
      setTimeout(tick, 8000);    // 启动 8 秒后先试一轮
      setTimeout(qqTick, 4000);  // 尽快确认 QQ 在线态
      console.log('[bridge] watching pet-events / memory-map / diary + qq probe ...');
    },
    stop() {
      state.stopped = true;
      if (state.timer) clearInterval(state.timer);
      if (state.qqTimer) clearInterval(state.qqTimer);
      state.timer = state.qqTimer = null;
    },
  };
}

module.exports = { createBridge };
