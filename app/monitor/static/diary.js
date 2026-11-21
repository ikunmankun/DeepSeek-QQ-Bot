/* 大肥鱼日记本：读 /api/diary 快照渲染近 30 天日记。只读，60 秒轮询。 */
(function () {
  'use strict';

  var statusChip = document.getElementById('status');
  var generatedAt = document.getElementById('generated');
  var statsBox = document.getElementById('stats');
  var dayList = document.getElementById('days');
  var reader = document.getElementById('reader');
  var emptyBox = document.getElementById('empty');
  var emptyTitle = document.getElementById('empty-title');
  var emptyReason = document.getElementById('empty-reason');
  var refreshButton = document.getElementById('refresh');

  var state = { days: [], selected: '' };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function setStatus(kind, text) {
    statusChip.className = 'chip' + (kind ? ' ' + kind : '');
    statusChip.textContent = text;
  }

  function tagClass(tag) {
    if (tag === '睡前记') return 'tag-chip reflection';
    if (tag === '逛B站') return 'tag-chip bili';
    return 'tag-chip';
  }

  function renderStats(stats) {
    statsBox.replaceChildren(
      stat('有日记的天', stats.days),
      stat('总字数', stats.total_chars),
      stat('睡前记', stats.reflection_days),
      stat('逛B站', stats.bili_count));
  }

  function stat(label, value) {
    var box = el('div', 'stat');
    box.appendChild(el('span', null, label));
    box.appendChild(el('strong', null, String(value === undefined || value === null ? 0 : value)));
    return box;
  }

  function renderDayList() {
    var nodes = state.days.map(function (day) {
      var item = el('button', 'day-item' + (day.date === state.selected ? ' active' : ''));
      item.type = 'button';
      var row = el('div', 'row');
      row.appendChild(el('span', 'date', day.date.slice(5)));
      row.appendChild(el('span', 'weekday', day.weekday || ''));
      item.appendChild(row);
      var meta = el('div', 'meta');
      meta.appendChild(el('span', null, day.entries.length + ' 条 · ' + day.total_chars + ' 字'));
      if (day.reflected) meta.appendChild(el('span', 'badge reflected', '睡前记'));
      if (day.bili_count) meta.appendChild(el('span', 'badge bili', 'B站×' + day.bili_count));
      item.appendChild(meta);
      item.addEventListener('click', function () {
        state.selected = day.date;
        render();
      });
      return item;
    });
    dayList.replaceChildren.apply(dayList, nodes.length ? nodes : [el('p', 'muted small', '还没有日记。')]);
  }

  function renderDay(day) {
    var box = el('div');
    var heading = el('div', 'day-heading');
    heading.appendChild(el('h2', null, day.date));
    heading.appendChild(el('span', 'weekday', (day.weekday || '') + ''));
    heading.appendChild(el('span', 'chars', day.entries.length + ' 条 · 共 ' + day.total_chars + ' 字'));
    box.appendChild(heading);
    day.entries.forEach(function (entry) {
      var card = el('article', 'entry' + (entry.reflection ? ' reflection' : ''));
      var meta = el('div', 'entry-meta');
      meta.appendChild(el('span', 'entry-time', entry.time));
      if (entry.tag) meta.appendChild(el('span', tagClass(entry.tag), entry.tag));
      if (entry.mood) meta.appendChild(el('span', 'tag-chip mood', '心情：' + entry.mood));
      meta.appendChild(el('span', 'muted small', entry.chars + ' 字'));
      card.appendChild(meta);
      card.appendChild(el('p', 'entry-text', entry.text));
      box.appendChild(card);
    });
    return box;
  }

  function render() {
    renderDayList();
    var selected = state.days.find(function (day) { return day.date === state.selected; });
    if (!selected) {
      reader.replaceChildren();
      emptyBox.hidden = false;
      emptyTitle.textContent = '这一天没有日记';
      return;
    }
    emptyBox.hidden = true;
    reader.replaceChildren(renderDay(selected));
  }

  function load() {
    setStatus('', '正在加载');
    fetch('/api/diary', { cache: 'no-store' }).then(function (resp) {
      return resp.json();
    }).then(function (data) {
      if (!data.available) {
        setStatus('error', data.reason === 'snapshot_missing' ? '快照未生成' : '快照不可读');
        emptyTitle.textContent = '日记本还没导出';
        emptyReason.textContent = '日记快照每 5 分钟导出一次，稍后刷新即可。';
        emptyBox.hidden = false;
        reader.replaceChildren();
        dayList.replaceChildren();
        statsBox.replaceChildren();
        return;
      }
      setStatus('ok', '在线');
      generatedAt.textContent = '导出于 ' + data.generated_at;
      state.days = data.days || [];
      if (!state.selected || !state.days.some(function (day) { return day.date === state.selected; })) {
        state.selected = state.days.length ? state.days[0].date : '';
      }
      renderStats(data.stats || {});
      render();
    }).catch(function () {
      setStatus('error', '连接失败');
    });
  }

  refreshButton.addEventListener('click', load);
  load();
  setInterval(load, 60000);
})();
