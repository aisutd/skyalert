/*
 * SkyAlert fire information panel.
 *
 * Runs inside the Folium map page (dashboard.py adds it). When a fire dot is
 * clicked, instead of Leaflet's plain popup it:
 *   1. rings the dot,
 *   2. places a flat card on a free spot of the map next to it,
 *   3. fills the card with real data from /api/fire-context (weather, place, fire-weather risk).
 *
 * Started by dashboard.py with: SkyAlertFirePanel.init(map, fireDotsLayer)
 */
(function () {
  'use strict';

  var PANEL_WIDTH = 360;      // px, on wide screens
  var GAP = 18;               // px between the dot and the panel
  var MARGIN = 16;            // px kept free around the map's edges
  var COMPACT_BELOW = 640;    // map narrower than this: panel docks to the bottom
  var SOUND_KEY = 'skyalert-sound';

  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  var map, dots;
  var fires = {};                  // "lat,lon" -> fire record from /api/fires
  var pathToLayer = new WeakMap(); // SVG <path> -> its Leaflet layer (for keyboard use)
  var safeTop = 0;                 // px hidden by the dashboard's stat cards
  var soundOn = false;
  var audio = null;

  // Everything about the current selection
  var current = { layer: null, record: null, context: null, trigger: null, request: 0, place: null };
  var panelState = 'closed';       // closed | open | closing
  var closeTimer = null;

  var root, panel, body, ring;

  // ---------- small helpers ----------

  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function keyOf(lat, lon) {
    return Number(lat).toFixed(5) + ',' + Number(lon).toFixed(5);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function motionMs(ms) {
    return reducedMotion.matches ? 0 : ms;
  }

  var CONFIDENCE = { l: 'Low', n: 'Nominal', h: 'High' };
  var COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];

  function compass(deg) {
    return COMPASS[Math.round(deg / 22.5) % 16];
  }

  function fmtCoord(lat, lon) {
    return Math.abs(lat).toFixed(4) + '° ' + (lat >= 0 ? 'N' : 'S') + '  ' +
           Math.abs(lon).toFixed(4) + '° ' + (lon >= 0 ? 'E' : 'W');
  }

  // FIRMS acq_time: 1942 -> "19:42"
  function fmtAcqTime(t) {
    var s = String(t).padStart(4, '0');
    return s.slice(0, 2) + ':' + s.slice(2);
  }

  // ---------- setup ----------

  function init(leafletMap, fireDots) {
    map = leafletMap;
    dots = fireDots;

    try { soundOn = localStorage.getItem(SOUND_KEY) === 'on'; } catch (e) { soundOn = false; }

    buildDom();
    addSoundControl();

    // Load the full fire records so the panel can show FRP, confidence and time
    fetch('/api/fires')
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (rows) {
        rows.forEach(function (f) { fires[keyOf(f.latitude, f.longitude)] = f; });
        dots.eachLayer(labelMarker); // refresh screen-reader labels with the full data
      })
      .catch(function () { /* the panel still works from the dot's position alone */ });

    dots.eachLayer(setupMarker);

    // Keyboard: Enter / Space on a focused fire dot opens it
    map.getPanes().overlayPane.addEventListener('keydown', function (e) {
      var layer = pathToLayer.get(e.target);
      if (layer && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        select(layer, true);
      }
    });

    map.on('zoomend', keepMarkerVisible);
    map.on('moveend', keepMarkerVisible);
    map.on('resize', function () { if (panelState === 'open') place(true); });
    // Clicking empty map closes the panel
    map.on('click', function () { close(); });
    // Switching to the heatmap hides the dots, so close too
    dots.on('remove', function () { close(true); });

    // The dashboard tells us how much of the top of the map its stat cards cover
    window.addEventListener('message', function (e) {
      if (e.origin !== window.location.origin) return;
      if (e.data && e.data.type === 'skyalert-safe-area') {
        safeTop = Math.max(0, Number(e.data.top) || 0);
        if (panelState === 'open') place(true);
      }
    });

    // Content height changes (loading -> loaded): keep the panel on screen
    if (window.ResizeObserver) {
      new ResizeObserver(function () {
        if (panelState === 'open') place(true);
      }).observe(panel);
    }
  }

  function setupMarker(layer) {
    if (!layer.getLatLng) return;
    // No default Leaflet popup for these dots
    layer.unbindPopup();
    // Don't let the click also reach the map (which would close the panel)
    layer.options.bubblingMouseEvents = false;
    layer.on('click', function (e) {
      L.DomEvent.stop(e);
      select(layer, false);
    });
    // Leaflet makes a fresh <path> each time the layer is shown again
    layer.on('add', function () { labelMarker(layer); });
    labelMarker(layer);
  }

  // Make a dot reachable with Tab and readable by screen readers
  function labelMarker(layer) {
    var path = layer.getElement && layer.getElement();
    if (!path) return;
    pathToLayer.set(path, layer);
    var ll = layer.getLatLng();
    var f = fires[keyOf(ll.lat, ll.lng)];
    path.setAttribute('tabindex', '0');
    path.setAttribute('role', 'button');
    path.setAttribute('aria-label', f
      ? 'Fire detected ' + f.acq_date + ' ' + fmtAcqTime(f.acq_time) + ' UTC, ' + f.frp + ' megawatts. Open details.'
      : 'Fire detection at ' + fmtCoord(ll.lat, ll.lng) + '. Open details.');
  }

  function buildDom() {
    var container = map.getContainer();
    root = L.DomUtil.create('div', 'fp-root', container);
    root.dataset.state = 'closed';

    panel = L.DomUtil.create('section', 'fp-panel', root);
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'false');
    panel.setAttribute('aria-labelledby', 'fp-title');
    panel.setAttribute('tabindex', '-1');
    panel.hidden = true;
    panel.innerHTML =
      '<div class="fp-content">' +
      '  <header class="fp-head">' +
      '    <span class="fp-eyebrow">Fire details</span>' +
      '    <button type="button" class="fp-close" aria-label="Close fire details">' +
      '      <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 3.5l9 9M12.5 3.5l-9 9"/></svg>' +
      '    </button>' +
      '  </header>' +
      '  <div class="fp-body"></div>' +
      '</div>';
    body = panel.querySelector('.fp-body');

    // Clicks and scrolling inside the panel shouldn't move or close the map
    L.DomEvent.disableClickPropagation(panel);
    L.DomEvent.disableScrollPropagation(panel);
    panel.querySelector('.fp-close').addEventListener('click', function () { close(); });
    panel.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        e.stopPropagation();
        close();
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && panelState === 'open') close();
    });
    body.addEventListener('click', function (e) {
      if (e.target.closest('.fp-analysis')) openAnalysis();
    });

    // The pulsing ring is a Leaflet marker, so it moves and zooms with the map by itself
    ring = L.marker([0, 0], {
      interactive: false,
      keyboard: false,
      icon: L.divIcon({
        className: 'fp-ring',
        iconSize: [0, 0],
        html: '<span></span>',
      }),
    });
  }

  // ---------- selecting / opening / closing ----------

  function select(layer, fromKeyboard) {
    if (layer === current.layer && panelState === 'open') return;
    var switching = panelState === 'open';
    var ll = layer.getLatLng();

    clearTimeout(closeTimer);
    current.layer = layer;
    current.record = fires[keyOf(ll.lat, ll.lng)] || { latitude: ll.lat, longitude: ll.lng };
    current.context = null;
    if (!switching) current.trigger = fromKeyboard ? layer.getElement() : null;

    showRing(ll);

    render('loading');

    if (switching) {
      place(true);
      sound('switch');
    } else {
      open();
      sound('open');
    }
    loadContext(current.record);
  }

  function open() {
    panelState = 'open';
    panel.hidden = false;
    root.dataset.state = 'closed';
    place(false);
    // Next frame: switch to "open" so the fade starts from the closed state
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        root.dataset.state = 'open';
        panel.focus({ preventScroll: true });
      });
    });
  }

  function close(immediately) {
    if (panelState !== 'open') return;
    panelState = 'closing';
    root.dataset.state = 'closing';
    current.request++; // ignore any data still on its way
    sound('close');

    var trigger = current.trigger;
    closeTimer = setTimeout(function () {
      panelState = 'closed';
      root.dataset.state = 'closed';
      panel.hidden = true;
      map.removeLayer(ring);
      current.layer = null;
      // Return keyboard focus to the dot that opened the panel
      if (trigger && document.contains(trigger)) trigger.focus({ preventScroll: true });
    }, immediately ? 0 : motionMs(180));
  }

  function showRing(latlng) {
    ring.setLatLng(latlng);
    if (!map.hasLayer(ring)) ring.addTo(map);
    var el = ring.getElement();
    el.classList.add('is-active');
  }

  // ---------- placement ----------

  // Pick a spot for the panel next to the dot, inside the map, clear of the stat cards
  function computePlace() {
    var size = map.getSize();
    var p = map.latLngToContainerPoint(current.layer.getLatLng());
    var top = safeTop + MARGIN;
    var compact = size.x < COMPACT_BELOW;
    var width = compact ? size.x - MARGIN * 2 : PANEL_WIDTH;
    var height = Math.min(panel.scrollHeight || 560, size.y - top - MARGIN);

    if (compact) {
      return { x: MARGIN, y: Math.max(top, size.y - height - MARGIN), width: width, height: height, side: 'bottom' };
    }

    var roomRight = size.x - p.x - GAP - MARGIN;
    var roomLeft = p.x - GAP - MARGIN;
    var side = roomRight >= width ? 'right' : roomLeft >= width ? 'left' : (roomRight >= roomLeft ? 'right' : 'left');
    var x = side === 'right' ? p.x + GAP : p.x - GAP - width;
    x = clamp(x, MARGIN, size.x - width - MARGIN);
    var y = clamp(p.y - height * 0.3, top, Math.max(top, size.y - height - MARGIN));
    return { x: x, y: y, width: width, height: height, side: side };
  }

  function place(animate) {
    var next = computePlace();
    current.place = next;
    panel.classList.toggle('is-compact', next.side === 'bottom');
    panel.classList.toggle('is-moving', !!animate && !reducedMotion.matches);
    panel.style.width = next.width + 'px';
    panel.style.maxHeight = (map.getSize().y - safeTop - MARGIN * 2) + 'px';
    panel.style.transform = 'translate3d(' + Math.round(next.x) + 'px,' + Math.round(next.y) + 'px,0)';
  }

  // If the dot ends up under the panel (or the docked bottom panel), slide the map a bit
  function keepMarkerVisible() {
    if (panelState !== 'open' || !current.place) return;
    var p = map.latLngToContainerPoint(current.layer.getLatLng());
    var r = current.place, pad = 24;
    var covered = p.x > r.x - pad && p.x < r.x + r.width + pad && p.y > r.y - pad && p.y < r.y + r.height + pad;
    if (!covered) return;
    if (r.side === 'bottom') {
      map.panBy([0, p.y - (r.y - 56)], { animate: !reducedMotion.matches });
    } else {
      place(true);
    }
  }

  // ---------- data ----------

  function loadContext(record) {
    var request = ++current.request;
    fetch('/api/fire-context?lat=' + record.latitude + '&lon=' + record.longitude)
      .then(function (r) {
        if (!r.ok) throw new Error('Server returned ' + r.status);
        return r.json();
      })
      .then(function (context) {
        if (request !== current.request) return; // a different dot was picked meanwhile
        current.context = context;
        render('ready');
      })
      .catch(function (err) {
        if (request !== current.request) return;
        current.context = { error: err.message };
        render('error');
      });
  }

  // Every block of the panel is an .fp-item
  function item(html) {
    return '<div class="fp-item">' + html + '</div>';
  }

  function skeleton(width) {
    return '<span class="fp-skel" style="width:' + width + '"></span>';
  }

  function render(status) {
    var f = current.record;
    var c = current.context || {};
    var w = c.weather;
    var loading = status === 'loading';
    var html = '';

    // Location
    var placeName = c.location ? c.location.name : loading ? null : 'Location unavailable';
    var country = c.location && c.location.country && c.location.country !== 'United States' ? ' · ' + c.location.country : '';
    html += item(
      '<h2 class="fp-title" id="fp-title">' + (placeName ? esc(placeName) : skeleton('70%')) + '</h2>' +
      '<div class="fp-coords">' + esc(fmtCoord(f.latitude, f.longitude)) + esc(country) + '</div>');

    // Risk level
    if (w) {
      var risk = w.risk;
      html += item(
        '<div class="fp-risk" data-risk="' + esc(risk.key) + '">' +
        '  <span class="fp-risk__dot" aria-hidden="true"></span>' +
        '  <div class="fp-risk__text">' +
        '    <div class="fp-label">Fire-weather risk</div>' +
        '    <div class="fp-risk__level">' + esc(risk.label) + '</div>' +
        '  </div>' +
        '  <div class="fp-risk__index" title="' + esc(risk.method) + '">' +
        '    <span class="fp-mono">' + risk.index.toFixed(0) + '</span><span class="fp-risk__max">/100</span>' +
        '  </div>' +
        '  <div class="fp-meter" aria-hidden="true"><span style="width:' + Math.max(2, risk.index) + '%"></span></div>' +
        '</div>');
    } else {
      html += item(loading ? skeleton('100%') : '<div class="fp-muted">Weather unavailable</div>');
    }

    // The FIRMS detection itself
    if (f.frp != null) {
      html += item(
        '<div class="fp-section-title">Satellite</div>' +
        '<div class="fp-stats fp-stats--3">' +
        stat('FRP', f.frp.toFixed(1) + ' MW') +
        stat('Confidence', CONFIDENCE[f.confidence] || esc(f.confidence), '', true) +
        stat('Seen', fmtAcqTime(f.acq_time) + ' UTC', esc(f.acq_date.slice(5)) + (f.daynight === 'N' ? ' · night' : ' · day')) +
        '</div>');
    }

    // Weather
    if (w) {
      var cur = w.current;
      html += item(
        '<div class="fp-section-title">Weather</div>' +
        '<div class="fp-stats fp-stats--3">' +
        stat('Temp', Math.round(cur.temperature_f) + '°F') +
        stat('Humidity', Math.round(cur.humidity) + '%') +
        stat('Wind', Math.round(cur.wind_mph) + ' mph', '', false, compass(cur.wind_direction)) +
        '</div>');
    } else if (loading) {
      html += item(skeleton('100%'));
    }

    html += item(
      '<button type="button" class="fp-analysis"' + (window.parent === window ? ' disabled' : '') + '>' +
      'Regional analysis <span aria-hidden="true">→</span></button>');

    body.innerHTML = html;
    body.setAttribute('aria-busy', loading ? 'true' : 'false');
  }

  // tag: short orange text on the right of the label, e.g. the wind direction
  function stat(label, value, sub, accent, tag) {
    return '<div class="fp-stat"><div class="fp-stat__head"><div class="fp-label">' + esc(label) + '</div>' +
           (tag ? '<span class="fp-stat__tag">' + esc(tag) + '</span>' : '') + '</div>' +
           '<div class="fp-stat__value fp-mono' + (accent ? ' fp-stat__value--accent' : '') + '">' + value + '</div>' +
           (sub ? '<div class="fp-stat__sub">' + esc(sub) + '</div>' : '') + '</div>';
  }

  // Tell the React dashboard to open the Analytics page around this fire
  function openAnalysis() {
    var f = current.record;
    var name = current.context && current.context.location ? current.context.location.name : fmtCoord(f.latitude, f.longitude);
    window.parent.postMessage(
      { type: 'skyalert-open-analysis', lat: f.latitude, lon: f.longitude, label: name },
      window.location.origin
    );
  }

  // ---------- sound (off by default) ----------

  function addSoundControl() {
    var SoundControl = L.Control.extend({
      options: { position: 'bottomright' },
      onAdd: function () {
        var wrap = L.DomUtil.create('div', 'leaflet-bar fp-sound');
        var btn = L.DomUtil.create('button', 'fp-sound__btn', wrap);
        btn.type = 'button';
        L.DomEvent.disableClickPropagation(wrap);
        function update() {
          btn.setAttribute('aria-pressed', soundOn ? 'true' : 'false');
          btn.title = soundOn ? 'Sound effects on' : 'Sound effects off';
          btn.setAttribute('aria-label', btn.title);
          btn.innerHTML = soundOn
            ? '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path d="M3 8h3l4-3v10l-4-3H3z"/><path d="M13 7.5a3.5 3.5 0 010 5M15 5.5a6.5 6.5 0 010 9" fill="none"/></svg>'
            : '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path d="M3 8h3l4-3v10l-4-3H3z"/><path d="M13.5 8l4 4M17.5 8l-4 4" fill="none"/></svg>';
        }
        btn.addEventListener('click', function () {
          soundOn = !soundOn;
          try { localStorage.setItem(SOUND_KEY, soundOn ? 'on' : 'off'); } catch (e) { /* not remembered */ }
          update();
          if (soundOn) sound('switch');
        });
        update();
        return wrap;
      },
    });
    // Added after the zoom buttons, so it sits just above them
    new SoundControl().addTo(map);
  }

  // Tiny synthesized tones: no audio files needed
  function sound(kind) {
    if (!soundOn) return;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === 'suspended') audio.resume();
      var now = audio.currentTime;
      var out = audio.createGain();
      out.gain.value = 0.045;
      out.connect(audio.destination);
      var tone = function (from, to, start, length, type) {
        var osc = audio.createOscillator();
        var env = audio.createGain();
        osc.type = type || 'sine';
        osc.frequency.setValueAtTime(from, now + start);
        osc.frequency.exponentialRampToValueAtTime(to, now + start + length);
        env.gain.setValueAtTime(0.0001, now + start);
        env.gain.exponentialRampToValueAtTime(1, now + start + 0.015);
        env.gain.exponentialRampToValueAtTime(0.0001, now + start + length);
        osc.connect(env).connect(out);
        osc.start(now + start);
        osc.stop(now + start + length + 0.02);
      };
      if (kind === 'open') { tone(520, 880, 0, 0.32); tone(1320, 1760, 0.12, 0.38); }
      else if (kind === 'switch') { tone(880, 1180, 0, 0.18, 'triangle'); }
      else if (kind === 'close') { tone(760, 420, 0, 0.26); }
    } catch (e) { /* audio not available; stay silent */ }
  }

  window.SkyAlertFirePanel = { init: init };
})();
