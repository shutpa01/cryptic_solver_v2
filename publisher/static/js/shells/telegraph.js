/* Telegraph shell — chrome and controls.
 *
 * Owns: the top bar (back, timer, plaque, Reveal / Check / More), where the
 * grid and the two clue columns sit, and the More options. Everything about
 * how a square behaves lives in engine.js; if a change belongs to solving
 * rather than to this paper's look, it goes there instead.
 */
(function () {
  'use strict';

  var C = window.Cordelia;
  var boot = window.CORDELIA_BOOT || {};
  var model = JSON.parse(document.getElementById('tg-model').textContent);

  var engine = new C.Engine(model);
  var api = new C.Api({
    token: boot.token,
    renewAfter: boot.renewAfter,
    onExpire: function () { status('Session expired — reload the page.', true); }
  });

  var grid = new C.GridView(engine, document.getElementById('tg-grid'));
  var bar = new C.ClueBarView(engine, document.getElementById('tg-bar'));
  var across = new C.ClueListView(engine, document.getElementById('tg-across'), C.ACROSS);
  var down = new C.ClueListView(engine, document.getElementById('tg-down'), C.DOWN);
  new C.MatchCount(engine, api, grid);

  var clueColumns = document.getElementById('tg-clue-columns');
  var tools = new C.ToolsView(engine, api, document.getElementById('tg-tools'), {
    onOpen: function () { clueColumns.hidden = true; },
    onClose: function () {
      clueColumns.hidden = false;
      bar.clearWords();
      grid.focus();
    }
  });

  // Right-click anywhere in the grid is the desktop way in. There is no
  // right-click on a phone, which is why the fourth icon exists as well.
  document.getElementById('tg-grid').addEventListener('contextmenu', function (event) {
    event.preventDefault();
    tools.show();
  });

  var statusEl = document.getElementById('tg-status');

  function status(message, warn) {
    statusEl.textContent = message || '';
    statusEl.classList.toggle('is-warn', !!warn);
  }

  /* --- timer -------------------------------------------------------- */

  var timerEl = document.getElementById('tg-timer');
  var seconds = 0;

  window.setInterval(function () {
    seconds += 1;
    var m = Math.floor(seconds / 60);
    var s = seconds % 60;
    timerEl.textContent = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }, 1000);

  /* --- menus -------------------------------------------------------- */

  function closeMenus(except) {
    document.querySelectorAll('.tg-dropdown').forEach(function (menu) {
      if (menu === except) return;
      menu.hidden = true;
    });
    document.querySelectorAll('.tg-tool').forEach(function (button) {
      button.classList.remove('is-open');
      button.setAttribute('aria-expanded', 'false');
    });
    if (except) {
      var owner = document.querySelector('.tg-tool[data-menu="' + except.dataset.for + '"]');
      owner.classList.add('is-open');
      owner.setAttribute('aria-expanded', 'true');
    }
  }

  document.querySelectorAll('.tg-tool').forEach(function (button) {
    button.addEventListener('click', function () {
      // The Tools icon opens the menu to choose a tool, and closes tools mode
      // again once it is open — one control in and out, as designed.
      if (button.dataset.menu === 'tools' && tools.isOpen()) {
        closeMenus(null);
        tools.hide();
        return;
      }
      var menu = document.querySelector('.tg-dropdown[data-for="' + button.dataset.menu + '"]');
      var opening = menu.hidden;
      closeMenus(opening ? menu : null);
      menu.hidden = !opening;
      button.setAttribute('aria-expanded', opening ? 'true' : 'false');
    });
  });

  // Close on a click outside any menu. This is decided by where the click
  // landed, NOT by stopping propagation inside the dropdowns — stopping it
  // there also kept clicks from reaching the delegated action handler below,
  // which left every Reveal and Check item doing nothing at all.
  document.addEventListener('click', function (event) {
    if (event.target.closest('.tg-menu')) return;
    closeMenus(null);
  });

  /* --- check and reveal --------------------------------------------- */

  function lettersFor(scope) {
    var out = {};
    if (scope === 'grid') {
      Object.keys(engine.letters).forEach(function (k) { out[k] = engine.letters[k]; });
      return out;
    }
    var entry = engine.current();
    if (!entry) return out;
    var cells = scope === 'letter' && engine.cursor
      ? [[engine.cursor.r, engine.cursor.c]]
      : entry.cells;
    cells.forEach(function (rc) {
      var k = C.key(rc[0], rc[1]);
      if (engine.letters[k]) out[k] = engine.letters[k];
    });
    return out;
  }

  function runCheck(scope) {
    var letters = lettersFor(scope);
    if (!Object.keys(letters).length) {
      status('Nothing to check yet.');
      return;
    }
    api.post('/api/check', { letters: letters, scope: 'grid' })
      .then(function (data) {
        engine.markWrong(data.wrong || []);
        var unknown = (data.unknown || []).length;
        if (data.wrong.length === 0 && !unknown) {
          status(scope === 'grid' ? 'All correct so far.' : 'Correct.');
        } else if (data.wrong.length) {
          status(data.wrong.length + ' letter' +
            (data.wrong.length === 1 ? '' : 's') + ' wrong.', true);
        }
        // A prize puzzle under embargo has no answers here. Say so rather
        // than let silence read as a pass.
        if (unknown) {
          status(unknown + ' square' + (unknown === 1 ? '' : 's') +
            ' cannot be checked — answers are not published for this puzzle yet.',
            true);
        }
      })
      .catch(function () { status('Could not check just now.', true); });
  }

  function runReveal(scope) {
    var body = { scope: scope };
    if (scope === 'cell' && engine.cursor) {
      body.cell = C.key(engine.cursor.r, engine.cursor.c);
    } else if (scope === 'entry') {
      var entry = engine.current();
      if (!entry) return;
      body.entry = entry.id;
    }
    api.post('/api/reveal', body)
      .then(function (data) {
        var letters = data.letters || {};
        if (!Object.keys(letters).length) {
          status('No answers are published for this puzzle yet.', true);
          return;
        }
        engine.applyLetters(letters);
        status(data.unavailable
          ? 'Revealed what is available — some answers are not published yet.'
          : '');
      })
      .catch(function () { status('Could not reveal just now.', true); });
  }

  /* --- options ------------------------------------------------------ */

  function setFont(size) {
    document.body.dataset.font = size;
    document.querySelectorAll('.tg-steps button').forEach(function (button) {
      button.classList.toggle('is-on', button.dataset.size === String(size));
    });
    store('font', size);
  }

  function store(name, value) {
    try {
      window.localStorage.setItem('cordelia_pub_opt_' + name, JSON.stringify(value));
    } catch (e) { /* private mode — options simply do not persist */ }
  }

  function stored(name, fallback) {
    try {
      var raw = window.localStorage.getItem('cordelia_pub_opt_' + name);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (e) { return fallback; }
  }

  function applyOption(name, on) {
    if (name === 'showTimer') {
      timerEl.hidden = !on;
    } else if (name === 'hideCompleted') {
      document.querySelectorAll('.cg-clues').forEach(function (list) {
        list.classList.toggle('hide-completed', on);
      });
      markCompleted();
    } else {
      engine.options[name] = on;
    }
    store(name, on);
  }

  document.querySelectorAll('[data-option]').forEach(function (input) {
    var name = input.dataset.option;
    input.checked = stored(name, input.checked);
    applyOption(name, input.checked);
    input.addEventListener('change', function () { applyOption(name, input.checked); });
  });

  setFont(stored('font', 2));

  /* --- actions ------------------------------------------------------ */

  var actions = {
    'reveal-letter': function () { runReveal('cell'); },
    'reveal-word': function () { runReveal('entry'); },
    'reveal-grid': function () { runReveal('grid'); },
    'check-letter': function () { runCheck('letter'); },
    'check-word': function () { runCheck('word'); },
    'check-grid': function () { runCheck('grid'); },
    'reset': function () {
      engine.reset();
      status('Puzzle reset.');
    }
  };

  document.addEventListener('click', function (event) {
    var button = event.target.closest('[data-action]');
    if (!button) return;
    if (button.dataset.action === 'font') {
      setFont(button.dataset.size);
      return;
    }
    if (button.dataset.action === 'tool') {
      closeMenus(null);
      tools.show(button.dataset.tab);
      return;
    }
    var action = actions[button.dataset.action];
    if (!action) return;
    closeMenus(null);
    action();
    grid.focus();
  });

  document.getElementById('tg-back').addEventListener('click', function () {
    // Inside a publisher's iframe there is nowhere of ours to go back to, so
    // tell the host page and let it decide. It is their navigation, not ours.
    window.parent.postMessage({ cordelia: 'back' }, '*');
  });

  /* --- completed clues --------------------------------------------- */

  function markCompleted() {
    model.entries.forEach(function (entry) {
      if (entry.stub_of || !entry.len) return;
      var done = engine.isFull(entry.id);
      document.querySelectorAll('.cg-clue[data-entry="' + entry.id + '"]')
        .forEach(function (item) { item.classList.toggle('is-done', done); });
    });
  }

  engine.on('letters', markCompleted);
  engine.on('progress', function (p) {
    if (p.filled === p.total && p.total) status('Grid complete.');
  });
  markCompleted();

  /* Auto check: mark a wrong letter as soon as an entry is filled. Off by
   * default, exactly as the paper ships it. */
  engine.on('letters', function () {
    if (!engine.options.autoCheck) return;
    var entry = engine.current();
    if (!entry || !engine.isFull(entry.id)) return;
    runCheck('word');
  });

  grid.focus();
  status('');
}());
