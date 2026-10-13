() => {
  // Visible, unoccluded DOM text in the 1280x720 viewport, one box per text-node line fragment.
  const W = window.innerWidth, H = window.innerHeight;
  const boxes = [];
  const stats = {nodes: 0, words: 0, partial: 0, occluded: 0, transparent: 0, tiny: 0, inputs: 0};
  const transform = (s, t) => t === 'uppercase' ? s.toUpperCase() : t === 'lowercase' ? s.toLowerCase()
    : t === 'capitalize' ? s.replace(/(^|\s)(\S)/g, (m, a, b) => a + b.toUpperCase()) : s;
  const alpha = (color) => {
    const m = color.match(/rgba?\(([^)]+)\)/);
    if (!m) return 1;
    const parts = m[1].split(/[ ,/]+/).filter(Boolean);
    return parts.length >= 4 ? parseFloat(parts[3]) : 1;
  };
  const visibleAt = (el, x, y) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return false;
    const hit = document.elementFromPoint(x, y);
    return !!hit && (hit === el || el.contains(hit) || hit.contains(el));
  };
  const unoccluded = (el, r) => {
    const y = r.top + r.height / 2;
    const xs = [r.left + r.width * 0.5, r.left + Math.min(r.width * 0.15, 6), r.right - Math.min(r.width * 0.15, 6)];
    return xs.every(x => visibleAt(el, x, y));
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  let node;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue;
    if (!text || !text.trim()) continue;
    const el = node.parentElement;
    if (!el || ['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'OPTION', 'TITLE'].includes(el.tagName)) continue;
    if (el.checkVisibility && !el.checkVisibility({opacityProperty: true, visibilityProperty: true, contentVisibilityAuto: true})) continue;
    const cs = getComputedStyle(el);
    if (alpha(cs.color) < 0.3 || parseFloat(cs.opacity) < 0.3) { stats.transparent++; continue; }
    const fontSize = parseFloat(cs.fontSize) || 0;
    stats.nodes++;
    const lines = [];
    const re = /\S+/g;
    let m;
    while ((m = re.exec(text))) {
      range.setStart(node, m.index);
      range.setEnd(node, m.index + m[0].length);
      const rects = Array.from(range.getClientRects()).filter(r => r.width > 0.5 && r.height > 0.5);
      if (!rects.length) continue;
      stats.words++;
      // Union of fragments (a word rarely spans lines; then treat it as its first fragment).
      const r = rects[0];
      if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) continue;
      if (r.left < 0 || r.top < 0 || r.right > W || r.bottom > H) { stats.partial++; lines.push({partial: true, r}); continue; }
      if (fontSize < 7) { stats.tiny++; }
      if (!unoccluded(el, r)) { stats.occluded++; continue; }
      const word = transform(m[0], cs.textTransform);
      const last = lines.length ? lines[lines.length - 1] : null;
      if (last && !last.partial && Math.abs(last.top - r.top) < Math.max(2, r.height * 0.3) && r.left >= last.right - 1 && r.left - last.right < Math.max(fontSize, 4) * 1.5) {
        last.words.push(word);
        last.right = Math.max(last.right, r.right); last.bottom = Math.max(last.bottom, r.bottom);
        last.top = Math.min(last.top, r.top); last.left = Math.min(last.left, r.left);
      } else {
        lines.push({words: [word], left: r.left, top: r.top, right: r.right, bottom: r.bottom});
      }
    }
    for (const l of lines) {
      if (l.partial) continue;
      boxes.push({text: l.words.join(' '), bbox: [l.left, l.top, l.right, l.bottom], source: 'text', tag: el.tagName.toLowerCase(),
                  font_size: fontSize, font_weight: cs.fontWeight, color: cs.color});
    }
  }
  // Form controls render their value or placeholder without text nodes.
  const canvas = document.createElement('canvas').getContext('2d');
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const type = (el.getAttribute('type') || 'text').toLowerCase();
    if (el.tagName === 'INPUT' && !['text', 'search', 'email', 'url', 'tel', 'submit', 'button', 'reset', 'number'].includes(type)) continue;
    if (el.checkVisibility && !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    let value = el.tagName === 'SELECT' ? (el.selectedOptions[0] ? el.selectedOptions[0].text : '') : (el.value || el.getAttribute('placeholder') || '');
    if ((type === 'submit' || type === 'reset') && !value) value = type === 'submit' ? 'Submit' : 'Reset';
    value = value.replace(/\s+/g, ' ').trim();
    if (!value || el.tagName === 'TEXTAREA' && value.length > 80) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4 || r.left < 0 || r.top < 0 || r.right > W || r.bottom > H) continue;
    const cs = getComputedStyle(el);
    canvas.font = cs.font;
    const width = canvas.measureText(value).width;
    const fontSize = parseFloat(cs.fontSize) || 12;
    const left0 = r.left + parseFloat(cs.borderLeftWidth) + parseFloat(cs.paddingLeft);
    const right0 = r.right - parseFloat(cs.borderRightWidth) - parseFloat(cs.paddingRight);
    if (width > right0 - left0 + 2) continue;  // scrolled/clipped value
    const centered = ['submit', 'button', 'reset'].includes(type) || cs.textAlign === 'center';
    const left = centered ? (left0 + right0 - width) / 2 : cs.textAlign === 'right' || cs.textAlign === 'end' ? right0 - width : left0;
    const cy = el.tagName === 'TEXTAREA' ? r.top + parseFloat(cs.borderTopWidth) + parseFloat(cs.paddingTop) + fontSize * 0.6 : r.top + r.height / 2;
    const box = [left, cy - fontSize * 0.6, left + width, cy + fontSize * 0.6];
    if (!unoccluded(el, {left: box[0], right: box[2], top: box[1], height: box[3] - box[1], width: width})) { stats.occluded++; continue; }
    stats.inputs++;
    boxes.push({text: value, bbox: box, source: el.tagName.toLowerCase(), tag: el.tagName.toLowerCase(), font_size: fontSize,
                font_weight: cs.fontWeight, color: cs.color});
  }
  // Imagery regions (text inside them is not DOM text): img/svg/canvas/video/iframe and background images.
  const imagery = [];
  for (const el of document.querySelectorAll('*')) {
    const tag = el.tagName;
    const media = tag === 'INPUT' ? el.type === 'image' : ['IMG', 'svg', 'SVG', 'CANVAS', 'VIDEO', 'IFRAME', 'PICTURE', 'OBJECT', 'EMBED'].includes(tag);
    if (!media) {
      const bg = getComputedStyle(el).backgroundImage;
      if (!bg || bg === 'none' || !bg.includes('url(')) continue;
    }
    if (el.checkVisibility && !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    const r = el.getBoundingClientRect();
    const x1 = Math.max(0, r.left), y1 = Math.max(0, r.top), x2 = Math.min(W, r.right), y2 = Math.min(H, r.bottom);
    if (x2 - x1 < 8 || y2 - y1 < 8) continue;
    imagery.push({bbox: [x1, y1, x2, y2], tag: tag.toLowerCase(), background: !media});
    if (imagery.length > 400) break;
  }
  const bodyStyle = getComputedStyle(document.body);
  return {boxes, stats, imagery, title: document.title, lang: document.documentElement.lang || '',
          scroll_height: document.documentElement.scrollHeight, body_bg: bodyStyle.backgroundColor,
          iframes: document.querySelectorAll('iframe').length, videos: document.querySelectorAll('video').length,
          canvases: document.querySelectorAll('canvas').length};
}
