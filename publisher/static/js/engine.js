/* Cordelia publisher widget — the engine.
 *
 * One engine, three shells. Everything here is shared by the Telegraph, Times
 * and Guardian shells: the grid model, selection, typing and keyboard
 * semantics, the crossing-letter match count, persistence, and the calls to
 * our endpoint. A shell owns layout, chrome and controls — and nothing else.
 *
 * That split is only cheap if it is made from the first line, so the rule is
 * strict: nothing in this file knows what a shell looks like. It renders class
 * hooks (cg-*) and emits events; the shell's stylesheet decides everything
 * visual, and the shell's script decides where the pieces sit on the page.
 */
(function (global) {
  'use strict';

  var ACROSS = 'across';
  var DOWN = 'down';

  function key(r, c) { return r + ',' + c; }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  /* ------------------------------------------------------------------ *
   * Engine — model and state. No DOM.
   * ------------------------------------------------------------------ */

  function Engine(model, options) {
    this.model = model;
    this.options = Object.assign({
      skipFilled: false,
      autoCheck: false,
      storageKey: 'cordelia_pub_' + model.source + '_' + model.number
    }, options || {});

    this.entries = {};
    // Which entries START at a square, by direction. A clue number labels the
    // entries that begin there, which is not the same as the entries running
    // through there — that difference is the whole point of clicking a number
    // rather than a square.
    this.startsAt = {};
    // Which entries RUN THROUGH a square. A square carrying two of them is a
    // crossing, and a crossing that has a letter in it is the only way the
    // grid can narrow an entry down — which is what decides whether an entry
    // still has anything to learn from the board (see crossingsFilled).
    this.through = {};
    var self = this;
    model.entries.forEach(function (entry) {
      self.entries[entry.id] = entry;
      if (entry.stub_of || !entry.cells.length) return;
      var first = entry.cells[0];
      var at = self.startsAt[key(first[0], first[1])] || {};
      at[entry.dir] = entry.id;
      self.startsAt[key(first[0], first[1])] = at;
      entry.cells.forEach(function (rc) {
        var k = key(rc[0], rc[1]);
        (self.through[k] = self.through[k] || []).push(entry.id);
      });
    });

    this.letters = {};        // "r,c" -> "A"
    this.wrong = {};          // "r,c" -> true, cleared as soon as it is retyped
    this.cursor = null;       // {r, c}
    this.dir = ACROSS;
    this.activeId = null;
    this._handlers = {};

    // Picked clue words, as indices into the ACTIVE clue's tokens. This lives
    // on the engine rather than in a view because the same selection shows in
    // three places at once — the clue bar, the active row of the clue list,
    // and the header of the tools panel. Two views each holding their own copy
    // would drift apart the moment a solver used both.
    this.picked = [];

    this.restore();
    var first = this.firstEntry();
    if (first) this.select(first.id);
  }

  Engine.prototype.on = function (event, fn) {
    (this._handlers[event] = this._handlers[event] || []).push(fn);
    return this;
  };

  Engine.prototype.emit = function (event, payload) {
    (this._handlers[event] || []).forEach(function (fn) { fn(payload); });
  };

  /* --- geometry ----------------------------------------------------- */

  Engine.prototype.cellAt = function (r, c) {
    var row = this.model.cells[r];
    return row ? (row[c] || null) : null;
  };

  Engine.prototype.entry = function (id) { return this.entries[id] || null; };

  Engine.prototype.current = function () { return this.entry(this.activeId); };

  Engine.prototype.firstEntry = function () {
    for (var i = 0; i < this.model.entries.length; i++) {
      var entry = this.model.entries[i];
      if (!entry.stub_of && entry.len) return entry;
    }
    return null;
  };

  /* A stub ("See 1 Across") has no squares of its own — it selects the entry
   * that owns them, which is what the paper's own list does. */
  Engine.prototype.resolve = function (id) {
    var entry = this.entry(id);
    if (entry && entry.stub_of) return this.entry(entry.stub_of);
    return entry;
  };

  Engine.prototype.entryIdAt = function (r, c, dir) {
    var cell = this.cellAt(r, c);
    if (!cell) return null;
    return dir === DOWN ? cell.d : cell.a;
  };

  Engine.prototype.indexInEntry = function (entry, r, c) {
    for (var i = 0; i < entry.cells.length; i++) {
      if (entry.cells[i][0] === r && entry.cells[i][1] === c) return i;
    }
    return -1;
  };

  /* --- selection ---------------------------------------------------- */

  Engine.prototype.select = function (id, cellIndex) {
    var entry = this.resolve(id);
    if (!entry || !entry.len) return;
    this.activeId = entry.id;
    this.dir = entry.dir;
    var index = cellIndex == null ? this.firstUnfilledIndex(entry) : cellIndex;
    var cell = entry.cells[Math.max(0, Math.min(index, entry.len - 1))];
    this.cursor = { r: cell[0], c: cell[1] };
    this._announce();
  };

  Engine.prototype.firstUnfilledIndex = function (entry) {
    for (var i = 0; i < entry.cells.length; i++) {
      if (!this.letters[key(entry.cells[i][0], entry.cells[i][1])]) return i;
    }
    return 0;
  };

  /* Clicking a square selects the entry running in the current direction.
   * Clicking the square you are already on flips direction — the standard
   * behaviour for a shared cell, and the only way to reach a down entry by
   * mouse alone. */
  Engine.prototype.selectCell = function (r, c) {
    var cell = this.cellAt(r, c);
    if (!cell) return;
    var same = this.cursor && this.cursor.r === r && this.cursor.c === c;
    var dir = this.dir;
    if (same) {
      var other = dir === ACROSS ? DOWN : ACROSS;
      if (this.entryIdAt(r, c, other)) dir = other;
    } else if (!this.entryIdAt(r, c, dir)) {
      dir = dir === ACROSS ? DOWN : ACROSS;
    }
    var id = this.entryIdAt(r, c, dir);
    if (!id) return;
    this.dir = dir;
    this.activeId = id;
    this.cursor = { r: r, c: c };
    this._announce();
  };

  /* Clicking the number in a square jumps straight to the clue it labels,
   * from wherever you are. Where a number labels both an across and a down
   * clue, it takes across first and swaps on a second click — the same
   * two-state behaviour as clicking a shared square, anchored to the number. */
  Engine.prototype.selectByNumber = function (r, c) {
    var starts = this.startsAt[key(r, c)];
    if (!starts) {
      this.selectCell(r, c);      // no clue begins here — behave as a square
      return;
    }
    var across = starts[ACROSS];
    var down = starts[DOWN];
    var target;
    if (across && down) {
      target = this.activeId === across ? down : across;
    } else {
      target = across || down;
    }
    if (target) this.select(target);
  };

  Engine.prototype.flip = function () {
    if (!this.cursor) return;
    var other = this.dir === ACROSS ? DOWN : ACROSS;
    var id = this.entryIdAt(this.cursor.r, this.cursor.c, other);
    if (!id) return;
    this.dir = other;
    this.activeId = id;
    this._announce();
  };

  /* Tab / the clue-bar chevrons: move through the list in reading order,
   * across then down, skipping stubs. */
  Engine.prototype.step = function (delta) {
    var solid = this.model.entries.filter(function (e) {
      return !e.stub_of && e.len;
    });
    if (!solid.length) return;
    var at = solid.findIndex(function (e) { return e.id === this.activeId; }, this);
    var next = (at + delta + solid.length) % solid.length;
    this.select(solid[next].id);
  };

  Engine.prototype.selectionState = function () {
    var entry = this.current();
    return {
      entry: entry,
      cursor: this.cursor,
      dir: this.dir,
      cells: entry ? entry.cells : []
    };
  };

  /* Every route to a new selection goes through here, so picked words can be
   * dropped exactly when the clue changes — the indices refer to the active
   * clue's tokens and mean nothing against a different clue. Moving the cursor
   * WITHIN an entry keeps them. */
  Engine.prototype._announce = function () {
    if (this.activeId !== this._announcedId) {
      this._announcedId = this.activeId;
      if (this.picked.length) {
        this.picked = [];
        this.emit('words', { indices: [], words: [], phrase: '' });
      }
    }
    this.emit('selection', this.selectionState());
  };

  /* --- picked clue words -------------------------------------------- */

  /* Split a clue into tokens, keeping the whitespace so a view can rebuild the
   * line exactly. Word tokens carry the index the selection is expressed in,
   * so every surface numbers them identically. */
  Engine.prototype.clueTokens = function (entry) {
    var tokens = [];
    var index = 0;
    (entry && entry.clue ? entry.clue : '').split(/(\s+)/).forEach(function (raw) {
      if (!raw) return;
      if (!raw.trim()) {
        tokens.push({ text: raw, word: false, index: null });
        return;
      }
      tokens.push({
        text: raw,
        word: true,
        index: index++,
        // Two forms, because the two tools need different things. `clean` is
        // letters only, which is what anagram fodder is. `plain` keeps the
        // word as the setter wrote it, punctuation and all: the reference
        // tables are keyed on a normalised form of the REAL word, so throwing
        // the apostrophe away in the browser is what lost "Jill's companion"
        // -> JACK. The server normalises; it cannot un-strip.
        clean: raw.replace(/[^A-Za-z]/g, ''),
        plain: raw.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, '')
      });
    });
    return tokens;
  };

  /* Adjacency selection: a run of neighbouring words, so "French kiss" can be
   * taken as a phrase. Clicking away from the run starts a new one rather than
   * building a nonsense scatter of words. */
  Engine.prototype.toggleWord = function (index) {
    var picked = this.picked.slice();
    var at = picked.indexOf(index);
    if (at !== -1) {
      // Only the ends can be dropped without splitting the run in two.
      if (index === picked[0] || index === picked[picked.length - 1]) {
        picked.splice(at, 1);
      } else {
        picked = [index];
      }
    } else if (!picked.length) {
      picked = [index];
    } else if (index === picked[0] - 1 || index === picked[picked.length - 1] + 1) {
      picked.push(index);
      picked.sort(function (a, b) { return a - b; });
    } else {
      picked = [index];
    }
    this.picked = picked;
    this.emit('words', {
      indices: picked.slice(),
      words: this.pickedWords(),
      phrase: this.pickedPhrase()
    });
  };

  Engine.prototype.pickedWords = function () {
    return this._pickedForm('clean');
  };

  /* The picked run as the clue writes it — the form the reference lookup wants. */
  Engine.prototype.pickedPhrase = function () {
    return this._pickedForm('plain').join(' ');
  };

  Engine.prototype._pickedForm = function (form) {
    var tokens = this.clueTokens(this.current()).filter(function (t) { return t.word; });
    return this.picked.map(function (i) {
      return tokens[i] ? tokens[i][form] : '';
    }).filter(Boolean);
  };

  Engine.prototype.clearWords = function () {
    if (!this.picked.length) return;
    this.picked = [];
    this.emit('words', { indices: [], words: [], phrase: '' });
  };

  /* --- shared clue rendering ---------------------------------------- */

  /* One renderer for all three places a clue appears with clickable words: the
   * bar under the grid, the active row of the clue list, and the tools header.
   * They must tokenise and number identically or a selection made in one would
   * highlight the wrong words in another. */
  function renderClueWords(engine, entry, container) {
    container.innerHTML = '';
    engine.clueTokens(entry).forEach(function (token) {
      if (!token.word) {
        container.appendChild(document.createTextNode(token.text));
        return;
      }
      var span = el('span', 'cg-word', token.text);
      span.dataset.index = token.index;
      span.addEventListener('click', function (event) {
        // The clue list row is itself a button that selects the entry; a word
        // click must not also re-trigger it.
        event.stopPropagation();
        engine.toggleWord(token.index);
      });
      container.appendChild(span);
    });
    paintPickedWords(engine, container);
  }

  /* The erase control for the active clue. Built here so the clue bar, the
   * active row of the clue list and the tools header all get the same button
   * rather than three near-identical ones. */
  function clearEntryButton(engine, entryId) {
    var button = el('button', 'cg-clear-entry', 'Clear');
    button.type = 'button';
    button.title = 'Erase this answer';
    button.setAttribute('aria-label', 'Erase this answer');
    button.addEventListener('click', function (event) {
      // The clue-list row is itself a selector; erasing must not read as a
      // click on the row.
      event.stopPropagation();
      engine.clearEntry(entryId);
    });
    button.addEventListener('mousedown', function (event) {
      event.stopPropagation();
    });
    return button;
  }

  function paintPickedWords(engine, root) {
    if (!root) return;
    root.querySelectorAll('.cg-word').forEach(function (span) {
      span.classList.toggle('is-picked',
        engine.picked.indexOf(Number(span.dataset.index)) !== -1);
    });
  }

  /* --- typing ------------------------------------------------------- */

  Engine.prototype.setLetter = function (r, c, letter) {
    var k = key(r, c);
    if (letter) this.letters[k] = letter.toUpperCase();
    else delete this.letters[k];
    delete this.wrong[k];
    this.persist();
    this.emit('letters', { cells: [k] });
    this.emit('progress', this.progress());
  };

  Engine.prototype.type = function (ch) {
    if (!this.cursor) return;
    this.setLetter(this.cursor.r, this.cursor.c, ch);
    this.advance(1);
  };

  Engine.prototype.backspace = function () {
    if (!this.cursor) return;
    var k = key(this.cursor.r, this.cursor.c);
    if (this.letters[k]) {
      this.setLetter(this.cursor.r, this.cursor.c, '');
      return;
    }
    this.advance(-1, true);
    if (this.cursor) this.setLetter(this.cursor.r, this.cursor.c, '');
  };

  /* Move along the current entry. `raw` ignores the skip-filled option, so
   * backspacing always lands on the previous square even when that square is
   * filled — otherwise you could never correct a finished word. */
  Engine.prototype.advance = function (delta, raw) {
    var entry = this.current();
    if (!entry || !this.cursor) return;
    var index = this.indexInEntry(entry, this.cursor.r, this.cursor.c);
    if (index < 0) return;
    var next = index + delta;
    while (next >= 0 && next < entry.len) {
      var cell = entry.cells[next];
      if (raw || !this.options.skipFilled || !this.letters[key(cell[0], cell[1])]) {
        this.cursor = { r: cell[0], c: cell[1] };
        this._announce();
        return;
      }
      next += delta;
    }
    // Ran off the end: stay put rather than jumping to another entry. The
    // paper's own solver does the same — the entry boundary is meaningful.
    this._announce();
  };

  Engine.prototype.moveCursor = function (dr, dc) {
    if (!this.cursor) return;
    var wantDir = dr !== 0 ? DOWN : ACROSS;
    if (this.dir !== wantDir && this.entryIdAt(this.cursor.r, this.cursor.c, wantDir)) {
      this.flip();
      return;
    }
    var r = this.cursor.r + dr;
    var c = this.cursor.c + dc;
    while (r >= 0 && r < this.model.rows && c >= 0 && c < this.model.cols) {
      if (this.cellAt(r, c)) {
        this.selectCellKeepingDirection(r, c);
        return;
      }
      r += dr;
      c += dc;
    }
  };

  Engine.prototype.selectCellKeepingDirection = function (r, c) {
    var id = this.entryIdAt(r, c, this.dir);
    if (!id) {
      this.selectCell(r, c);
      return;
    }
    this.activeId = id;
    this.cursor = { r: r, c: c };
    this._announce();
  };

  Engine.prototype.handleKey = function (event) {
    var k = event.key;
    if (event.ctrlKey || event.metaKey || event.altKey) return false;
    if (/^[a-zA-Z]$/.test(k)) { this.type(k); return true; }
    switch (k) {
      case 'Backspace': this.backspace(); return true;
      case 'Delete':
        if (this.cursor) this.setLetter(this.cursor.r, this.cursor.c, '');
        return true;
      case 'ArrowUp': this.moveCursor(-1, 0); return true;
      case 'ArrowDown': this.moveCursor(1, 0); return true;
      case 'ArrowLeft': this.moveCursor(0, -1); return true;
      case 'ArrowRight': this.moveCursor(0, 1); return true;
      case 'Tab': this.step(event.shiftKey ? -1 : 1); return true;
      case 'Enter': case ' ': this.flip(); return true;
      default: return false;
    }
  };

  /* --- patterns, progress, persistence ------------------------------ */

  Engine.prototype.patternFor = function (id) {
    var entry = this.resolve(id);
    if (!entry) return '';
    var letters = this.letters;
    return entry.cells.map(function (rc) {
      return letters[key(rc[0], rc[1])] || '?';
    }).join('');
  };

  Engine.prototype.isFull = function (id) {
    return this.patternFor(id).indexOf('?') === -1;
  };

  /* True when every square this entry shares with another entry already has a
   * letter in it.
   *
   * Such an entry has taken everything the grid can give it: no crossing clue
   * can add a letter, so its pattern will not narrow again however much more
   * of the puzzle is solved. That is the one case where the widget offers a
   * shortlist even though far more than nine words fit — otherwise those
   * clues stay silent forever, which is precisely where a solver is stuck. */
  Engine.prototype.crossingsFilled = function (id) {
    var entry = this.resolve(id);
    if (!entry || !entry.cells.length) return false;
    var self = this;
    var crossings = 0;
    var filled = 0;
    entry.cells.forEach(function (rc) {
      var k = key(rc[0], rc[1]);
      var here = self.through[k] || [];
      if (here.length < 2) return;          // nothing crosses this square
      crossings++;
      if (self.letters[k]) filled++;
    });
    return crossings > 0 && filled === crossings;
  };

  Engine.prototype.progress = function () {
    var total = 0;
    var filled = 0;
    for (var r = 0; r < this.model.rows; r++) {
      for (var c = 0; c < this.model.cols; c++) {
        if (!this.cellAt(r, c)) continue;
        total++;
        if (this.letters[key(r, c)]) filled++;
      }
    }
    return { filled: filled, total: total };
  };

  Engine.prototype.persist = function () {
    try {
      global.localStorage.setItem(this.options.storageKey,
        JSON.stringify({ letters: this.letters, saved: Date.now() }));
    } catch (e) { /* private mode, quota — solving must continue regardless */ }
  };

  Engine.prototype.restore = function () {
    try {
      var raw = global.localStorage.getItem(this.options.storageKey);
      if (raw) this.letters = JSON.parse(raw).letters || {};
    } catch (e) { this.letters = {}; }
  };

  /* Erase just this entry's letters. Crossing squares go with it — they belong
   * to this entry too, and leaving them would make "clear" a half-measure the
   * solver then has to finish by hand. */
  Engine.prototype.clearEntry = function (id) {
    var entry = this.resolve(id || this.activeId);
    if (!entry || !entry.cells.length) return;
    var cleared = [];
    var self = this;
    entry.cells.forEach(function (rc) {
      var k = key(rc[0], rc[1]);
      if (self.letters[k] === undefined && !self.wrong[k]) return;
      delete self.letters[k];
      delete self.wrong[k];
      cleared.push(k);
    });
    if (!cleared.length) return;
    this.persist();
    this.emit('letters', { cells: cleared });
    this.emit('progress', this.progress());

    // Put the caret back at the start of what was just emptied. Without this
    // the cursor stays wherever it was — usually the end — so clearing and
    // retyping drops the new letters into the last squares instead of the
    // first, which is never what "clear and start again" means.
    if (entry.id === this.activeId) {
      var first = entry.cells[0];
      this.cursor = { r: first[0], c: first[1] };
      this._announce();
    }
  };

  Engine.prototype.reset = function () {
    this.letters = {};
    this.wrong = {};
    this.persist();
    this.emit('letters', { cells: null });
    this.emit('progress', this.progress());
  };

  Engine.prototype.markWrong = function (keys) {
    var self = this;
    keys.forEach(function (k) { self.wrong[k] = true; });
    this.emit('letters', { cells: keys });
  };

  Engine.prototype.applyLetters = function (map) {
    var self = this;
    Object.keys(map).forEach(function (k) {
      self.letters[k] = map[k];
      delete self.wrong[k];
    });
    this.persist();
    this.emit('letters', { cells: Object.keys(map) });
    this.emit('progress', this.progress());
  };

  /* ------------------------------------------------------------------ *
   * Api — everything served from our endpoint.
   * ------------------------------------------------------------------ */

  function Api(options) {
    this.token = options.token;
    this.renewAfter = options.renewAfter || 900;
    this.onExpire = options.onExpire || function () {};
    this._scheduleRenew();
  }

  Api.prototype._scheduleRenew = function () {
    var self = this;
    global.clearTimeout(this._renewTimer);
    this._renewTimer = global.setTimeout(function () { self.renew(); },
      this.renewAfter * 1000);
  };

  Api.prototype.renew = function () {
    var self = this;
    return fetch('/api/token/renew', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: this.token })
    }).then(function (response) {
      if (!response.ok) throw new Error('renew failed');
      return response.json();
    }).then(function (data) {
      self.token = data.token;
      self.renewAfter = data.renew_after || self.renewAfter;
      self._scheduleRenew();
    }).catch(function () {
      self.onExpire();
    });
  };

  Api.prototype.post = function (path, body) {
    var self = this;
    return fetch(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + this.token
      },
      body: JSON.stringify(body || {})
    }).then(function (response) {
      if (response.status === 401) {
        // Expired mid-solve: renew once, then replay the call.
        return self.renew().then(function () { return self.post(path, body); });
      }
      if (!response.ok) throw new Error(path + ' -> ' + response.status);
      return response.json();
    });
  };

  /* ------------------------------------------------------------------ *
   * GridView — the squares. Shared by all three shells.
   * ------------------------------------------------------------------ */

  function GridView(engine, container) {
    this.engine = engine;
    this.container = container;
    this.cells = {};
    this.render();
    this.bind();

    var self = this;
    engine.on('selection', function () { self.paintSelection(); });
    engine.on('letters', function (payload) { self.paintLetters(payload.cells); });
  }

  GridView.prototype.render = function () {
    var model = this.engine.model;
    var table = el('div', 'cg-grid');
    table.style.setProperty('--cg-cols', model.cols);
    table.setAttribute('role', 'grid');

    for (var r = 0; r < model.rows; r++) {
      for (var c = 0; c < model.cols; c++) {
        var cell = this.engine.cellAt(r, c);
        if (!cell) {
          table.appendChild(el('div', 'cg-block'));
          continue;
        }
        var node = el('div', 'cg-cell');
        node.dataset.r = r;
        node.dataset.c = c;
        node.setAttribute('role', 'gridcell');
        if (cell.n) {
          var num = el('span', 'cg-num', cell.n);
          var starts = this.engine.startsAt[key(r, c)] || {};
          var labels = [];
          if (starts[ACROSS]) labels.push(cell.n + ' Across');
          if (starts[DOWN]) labels.push(cell.n + ' Down');
          if (labels.length) num.title = 'Go to ' + labels.join(' / ');
          node.appendChild(num);
        }
        node.appendChild(el('span', 'cg-letter'));
        // Two slots, because a square can be the LAST of an across entry and
        // the last of a down entry at once — the across count reads top-right
        // and the down count bottom-left, so neither hides the other, and
        // neither goes near the top-left, which the clue number owns.
        node.appendChild(el('span', 'cg-count cg-count-a'));
        node.appendChild(el('span', 'cg-count cg-count-d'));
        table.appendChild(node);
        this.cells[key(r, c)] = node;
      }
    }

    // One offscreen input owns the caret. Without it a tap on a square raises
    // no keyboard at all on a phone, which is most of the readership.
    var input = el('input', 'cg-capture');
    input.setAttribute('autocomplete', 'off');
    input.setAttribute('autocorrect', 'off');
    input.setAttribute('autocapitalize', 'characters');
    input.setAttribute('spellcheck', 'false');
    input.setAttribute('aria-label', 'Crossword entry');

    this.container.innerHTML = '';
    this.container.appendChild(table);
    this.container.appendChild(input);
    this.input = input;
    this.table = table;
    this.paintLetters(null);
    this.paintSelection();
  };

  GridView.prototype.bind = function () {
    var self = this;

    this.table.addEventListener('mousedown', function (event) {
      // LEFT button only. A right-click was running this whole path: it moved
      // the cursor (and flipped direction when it landed on the square you
      // were already on) and called preventDefault, which in some browsers
      // also suppresses the contextmenu event that opens tools mode. Right-
      // click should open the tools where you are, not move you first.
      if (event.button !== 0) return;
      var node = event.target.closest('.cg-cell');
      if (!node) return;
      event.preventDefault();      // keep focus on the capture input
      // The count is a CONTROL, exactly as on the live site: it says how many
      // words fit, and clicking it shows you which ones (puzzle.html:226 ->
      // patternFromCrossing, puzzle2.js:1179 — open the pattern finder on this
      // entry's pattern and run the search). The shell decides where that
      // lands, so the engine only reports the click.
      // It must never fall through to the square: the chip is drawn proud of
      // its cell, so the overhanging edge used to select the square BEHIND it
      // (a different clue), and on the cursor's own square a click flipped
      // direction. Either way the count vanished, since it only renders on the
      // active entry — "a dead click and then the number disappeared".
      var chip = event.target.closest('.cg-count');
      if (chip) {
        // The chip speaks for ONE entry, which is not necessarily the selected
        // one — every entry shows its count, so a click has to go to the clue
        // whose number was clicked.
        if (chip.dataset.entry) self.engine.select(chip.dataset.entry);
        if (self.onCountClick) self.onCountClick(self.engine.current());
        return;
      }
      var r = Number(node.dataset.r);
      var c = Number(node.dataset.c);
      // Handled here rather than with a listener on the number itself: the
      // square's own mousedown would otherwise fire first and move the
      // selection before the number's click ever arrived.
      if (event.target.closest('.cg-num')) self.engine.selectByNumber(r, c);
      else self.engine.selectCell(r, c);
      self.focus();
    });

    // Keys are taken at the document, not at the capture input. Focus is
    // easily lost — a click on the page background, or on Reveal/Check/More —
    // and a solver who then types at a grid that ignores them thinks the
    // puzzle has broken. Anything typed into a genuine control is left alone.
    document.addEventListener('keydown', function (event) {
      var target = event.target;
      if (target !== self.input && target.closest &&
          target.closest('input, textarea, select, button, [contenteditable]')) {
        return;
      }
      if (self.engine.handleKey(event)) event.preventDefault();
    });

    // Android soft keyboards frequently deliver no usable keydown — the letter
    // only shows up as an input event. Read it from there and clear the field.
    this.input.addEventListener('input', function () {
      var value = self.input.value;
      self.input.value = '';
      var letters = value.replace(/[^a-zA-Z]/g, '');
      for (var i = 0; i < letters.length; i++) self.engine.type(letters[i]);
    });
  };

  GridView.prototype.focus = function () {
    try { this.input.focus({ preventScroll: true }); } catch (e) { this.input.focus(); }
  };

  GridView.prototype.paintLetters = function (keys) {
    var engine = this.engine;
    var targets = keys || Object.keys(this.cells);
    targets.forEach(function (k) {
      var node = this.cells[k];
      if (!node) return;
      node.querySelector('.cg-letter').textContent = engine.letters[k] || '';
      node.classList.toggle('is-wrong', !!engine.wrong[k]);
    }, this);
  };

  GridView.prototype.paintSelection = function () {
    var engine = this.engine;
    var entry = engine.current();
    var active = {};
    if (entry) entry.cells.forEach(function (rc) { active[key(rc[0], rc[1])] = true; });
    var cursorKey = engine.cursor ? key(engine.cursor.r, engine.cursor.c) : null;

    Object.keys(this.cells).forEach(function (k) {
      var node = this.cells[k];
      node.classList.toggle('is-active', !!active[k]);
      node.classList.toggle('is-cursor', k === cursorKey);
    }, this);
  };

  /* The match count sits in the last square of an entry, and EVERY entry that
   * has one shows it at once — that is the point of the feature. The solver
   * scans the board, sees which clues are down to a handful of possibilities,
   * and goes at those. A count on the selected clue only would tell you nothing
   * until you had already chosen what to work on.
   *
   * A number here is never above nine, and the server sends nothing at all for
   * the entries where more than nine fit — so a chip means "this one has come
   * down to a handful", and the board thins out as it is solved rather than
   * carrying a number everywhere. The number is the length of the list the
   * chip opens, so the two can never contradict each other.
   *
   * Zero is red: the letters in the grid are wrong. A full entry gets no count
   * at all (refused server-side), so it can never be read as a free Check. */
  GridView.prototype.showCounts = function (results) {
    this.clearCounts();
    var self = this;
    (results || []).forEach(function (item) {
      var entry = item.entry;
      if (!entry || !entry.len || item.n == null) return;
      var last = entry.cells[entry.len - 1];
      var node = self.cells[key(last[0], last[1])];
      if (!node) return;
      var slot = node.querySelector(
        entry.dir === DOWN ? '.cg-count-d' : '.cg-count-a');
      if (!slot) return;

      slot.textContent = String(item.n);
      slot.classList.toggle('is-zero', item.n === 0);
      // The chip is a control, so it carries the entry it speaks for: clicking
      // it selects THAT clue and opens the words that fit, whichever clue you
      // were on before.
      slot.dataset.entry = entry.id;
      slot.title = (entry.number || '') + ' ' +
        (entry.dir === DOWN ? 'Down' : 'Across') +
        ' — open in pattern finder, see the words that fit';
      node.classList.add('has-count');
    });
  };

  GridView.prototype.clearCounts = function () {
    Object.keys(this.cells).forEach(function (k) {
      var node = this.cells[k];
      node.classList.remove('has-count');
      node.querySelectorAll('.cg-count').forEach(function (slot) {
        slot.textContent = '';
        slot.removeAttribute('title');
        delete slot.dataset.entry;
        slot.classList.remove('is-zero');
      });
    }, this);
  };

  /* ------------------------------------------------------------------ *
   * MatchCount — wires the engine's letters to the served count.
   * ------------------------------------------------------------------ */

  function MatchCount(engine, api, gridView, delay) {
    this.engine = engine;
    this.api = api;
    this.gridView = gridView;
    this.delay = delay == null ? 180 : delay;
    this.timer = null;

    var self = this;
    // Counts depend on the LETTERS, not on what is selected — the whole board
    // is counted, so moving between clues changes nothing and costs nothing.
    engine.on('letters', function () { self.schedule(); });
    this.schedule();
  }

  MatchCount.prototype.schedule = function () {
    var self = this;
    global.clearTimeout(this.timer);
    this.timer = global.setTimeout(function () { self.fetchNow(); }, this.delay);
  };

  /* One request for the whole grid. The endpoint takes up to 100 patterns at a
   * time and was built for exactly this; asking entry by entry was the narrow
   * half. Counts are NOT cleared before the reply lands — repainting them from
   * a fresh set avoids the board going blank on every keystroke. */
  MatchCount.prototype.fetchNow = function () {
    var self = this;
    var engine = this.engine;
    var patterns = {};
    var crossed = [];
    var wanted = [];
    engine.model.entries.forEach(function (entry) {
      if (entry.stub_of || !entry.len) return;
      if (engine.isFull(entry.id)) return;   // a full entry is refused a count
      patterns[entry.id] = engine.patternFor(entry.id);
      // Only the browser knows the shape of the grid, so it is the browser
      // that reports which entries can no longer be narrowed by their
      // crossings. The server decides what to do about it.
      if (engine.crossingsFilled(entry.id)) crossed.push(entry.id);
      wanted.push(entry);
    });
    if (!wanted.length) {
      this.gridView.clearCounts();
      return;
    }
    var stamp = ++this._seq;
    this.api.post('/api/match-counts', { patterns: patterns, crossed: crossed })
      .then(function (data) {
        if (stamp !== self._seq) return;     // a later keystroke already won
        var counts = data.counts || {};
        self.gridView.showCounts(wanted.map(function (entry) {
          var result = counts[entry.id];
          return {
            entry: entry,
            n: result ? result.n : null,
            capped: result ? result.capped : false
          };
        }));
      })
      .catch(function () { /* a count is an aid, never an interruption */ });
  };

  MatchCount.prototype._seq = 0;

  /* ------------------------------------------------------------------ *
   * ClueListView / ClueBarView — same markup everywhere, placed by the shell.
   * ------------------------------------------------------------------ */

  function ClueListView(engine, container, direction) {
    this.engine = engine;
    this.container = container;
    this.direction = direction;
    this.items = {};
    this.texts = {};
    this.wordedId = null;   // the row currently rendered with clickable words
    this.render();

    var self = this;
    engine.on('selection', function () { self.paint(); });
    engine.on('words', function () {
      if (self.wordedId) paintPickedWords(engine, self.texts[self.wordedId]);
    });
  }

  ClueListView.prototype.render = function () {
    var list = el('ol', 'cg-clues');
    var self = this;
    this.engine.model.entries.forEach(function (entry) {
      if (entry.dir !== self.direction) return;
      var item = el('li', 'cg-clue');
      item.dataset.entry = entry.id;
      item.appendChild(el('span', 'cg-clue-num', entry.number));
      var body = el('span', 'cg-clue-body');
      body.appendChild(el('span', 'cg-clue-text', entry.clue));
      if (entry.enum) body.appendChild(el('span', 'cg-clue-enum', '(' + entry.enum + ')'));
      item.appendChild(body);
      self.texts[entry.id] = item.querySelector('.cg-clue-text');
      // A clue is a control, so it has to behave like one: reachable by tab,
      // announced as a button, and activated by Enter or Space. Without this
      // the clue list is mouse-only.
      item.setAttribute('role', 'button');
      item.setAttribute('tabindex', '0');
      item.setAttribute('aria-label', entry.number + ' ' +
        (entry.dir === ACROSS ? 'Across' : 'Down') + ', ' + entry.clue +
        (entry.enum ? ', ' + entry.enum + ' letters' : ''));
      item.addEventListener('click', function () { self.engine.select(entry.id); });
      item.addEventListener('keydown', function (event) {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        self.engine.select(entry.id);
      });
      list.appendChild(item);
      self.items[entry.id] = item;
    });
    this.container.innerHTML = '';
    this.container.appendChild(list);
    this.paint();
  };

  ClueListView.prototype.paint = function () {
    var entry = this.engine.current();
    var activeId = entry ? entry.id : null;
    Object.keys(this.items).forEach(function (id) {
      var resolved = this.engine.resolve(id);
      var on = !!activeId && resolved && resolved.id === activeId;
      this.items[id].classList.toggle('is-active', on);
    }, this);
    this.paintWords(activeId, entry);
    if (activeId && this.items[activeId]) this.scrollTo(this.items[activeId]);
  };

  /* Clue words are clickable on the ACTIVE row only.
   *
   * On desktop this list is far more prominent than the bar under the grid, so
   * a bar-only interaction was never going to be found. The reason words were
   * kept out of the list was that a row is also the entry selector and a tap
   * meant for 14 Across would land on a word — which only applies to rows that
   * are not yet selected. Restricting the spans to the active row removes the
   * collision: an inactive row has no word spans to intercept the click, and
   * clicking the active row again would have done nothing anyway.
   */
  ClueListView.prototype.paintWords = function (activeId, entry) {
    if (this.wordedId === activeId) {
      if (activeId) paintPickedWords(this.engine, this.texts[activeId]);
      return;
    }
    // Put the previously active row back to plain text, and back to being a
    // button. A row keeps role="button" only while it is a plain selector;
    // once it carries clickable words, nesting them inside a button would be
    // both invalid and unusable with a screen reader.
    if (this.wordedId && this.texts[this.wordedId]) {
      var previous = this.engine.entry(this.wordedId);
      this.texts[this.wordedId].textContent = previous ? previous.clue : '';
      var previousItem = this.items[this.wordedId];
      if (previousItem) {
        previousItem.setAttribute('role', 'button');
        previousItem.setAttribute('tabindex', '0');
      }
    }
    if (this.clearButton && this.clearButton.parentNode) {
      this.clearButton.parentNode.removeChild(this.clearButton);
      this.clearButton = null;
    }
    this.wordedId = activeId;
    if (activeId && this.texts[activeId] && entry) {
      renderClueWords(this.engine, entry, this.texts[activeId]);
      var item = this.items[activeId];
      if (item) {
        item.removeAttribute('role');
        item.removeAttribute('tabindex');
        // The erase control appears on the highlighted row only — 32 of them
        // down a clue list would be noise, and only one clue is ever current.
        this.clearButton = clearEntryButton(this.engine, activeId);
        item.appendChild(this.clearButton);
      }
    }
  };

  ClueListView.prototype.scrollTo = function (item) {
    var pane = this.container.closest('.cg-scroll') || this.container;
    var top = item.offsetTop - pane.offsetTop;
    if (top < pane.scrollTop || top + item.offsetHeight > pane.scrollTop + pane.clientHeight) {
      pane.scrollTop = top - 12;
    }
  };

  function ClueBarView(engine, container) {
    this.engine = engine;
    this.container = container;
    this.render();
    var self = this;
    engine.on('selection', function () { self.paint(); });
    engine.on('words', function () { self.paintWords(); });
  }

  ClueBarView.prototype.render = function () {
    var self = this;
    this.container.innerHTML = '';
    this.container.classList.add('cg-bar');

    var prev = el('button', 'cg-bar-nav', '‹');
    prev.setAttribute('aria-label', 'Previous clue');
    prev.addEventListener('click', function () { self.engine.step(-1); });

    var next = el('button', 'cg-bar-nav', '›');
    next.setAttribute('aria-label', 'Next clue');
    next.addEventListener('click', function () { self.engine.step(1); });

    this.body = el('div', 'cg-bar-body');
    this.container.appendChild(prev);
    this.container.appendChild(this.body);
    this.container.appendChild(next);
    this.paint();
  };

  ClueBarView.prototype.paint = function () {
    var entry = this.engine.current();
    this.body.innerHTML = '';
    if (!entry) return;
    var label = entry.number + ' ' + (entry.dir === ACROSS ? 'Across' : 'Down');
    this.body.appendChild(el('span', 'cg-bar-num', label));

    var text = el('span', 'cg-bar-text');
    renderClueWords(this.engine, entry, text);
    this.body.appendChild(text);

    if (entry.enum) {
      this.body.appendChild(el('span', 'cg-bar-enum', '(' + entry.enum + ')'));
    }
    this.body.appendChild(clearEntryButton(this.engine, entry.id));
  };

  ClueBarView.prototype.paintWords = function () {
    paintPickedWords(this.engine, this.body);
  };

  ClueBarView.prototype.clearWords = function () {
    this.engine.clearWords();
  };

  /* ------------------------------------------------------------------ *
   * ToolsView — the working section that replaces the clue columns.
   *
   * Tabs across anagram, pattern, synonym, word lookup and hints, so a solver
   * moves between them without going back to the menu. It stays open until
   * closed; clues are chosen by clicking squares in the grid meanwhile.
   * ------------------------------------------------------------------ */

  var TABS = [
    { id: 'anagram', label: 'Anagram' },
    { id: 'pattern', label: 'Pattern' },
    { id: 'synonym', label: 'Synonym' },
    { id: 'lookup', label: 'Word' },
    { id: 'hints', label: 'Hints' }
  ];

  function ToolsView(engine, api, container, options) {
    this.engine = engine;
    this.api = api;
    this.container = container;
    this.options = options || {};
    this.tab = 'pattern';
    this.open = false;
    this.words = [];
    this.render();

    var self = this;
    this.entryId = null;
    engine.on('selection', function () {
      if (!self.open) return;
      var entry = engine.current();
      var id = entry ? entry.id : null;
      // A different clue means every field in the panel now belongs to the
      // wrong one: anagram fodder picked from the old clue, a pattern from the
      // old entry, a lookup of a word that is no longer on screen, and — worst
      // — hint rungs already revealed for the previous answer. Rebuild the
      // panel from scratch rather than leave any of it behind.
      if (id !== self.entryId) {
        self.entryId = id;
        self.words = [];
        self.renderPanel();
      }
      self.refreshHeader();
    });
    engine.on('letters', function () {
      if (self.open && self.tab === 'pattern') self.prefillPattern();
    });
    engine.on('words', function (payload) {
      self.words = payload.words;
      self.phrase = payload.phrase || payload.words.join(' ');
      paintPickedWords(engine, self.headerBody);
      // An empty payload is the selection being cleared as the clue changed.
      // Opening a tool panel off the back of that would yank the working
      // section around every time the solver moved to another clue.
      if (!self.words.length) return;
      if (self.open && self.tab === 'anagram') {
        self.runAnagram();
        return;
      }
      self.show('lookup');
      self.runLookup();
    });
  }

  ToolsView.prototype.render = function () {
    var self = this;
    this.container.classList.add('cg-tools');
    this.container.hidden = true;
    this.container.innerHTML = '';

    // The current clue lives at the TOP of the working section with chevrons.
    // Not borrowed from the Telegraph's clue bar: the Times and Guardian have
    // no such bar, and one behaviour across all three shells is the point.
    var header = el('div', 'cg-tools-head');
    var prev = el('button', 'cg-bar-nav', '‹');
    prev.setAttribute('aria-label', 'Previous clue');
    prev.addEventListener('click', function () { self.engine.step(-1); });
    var next = el('button', 'cg-bar-nav', '›');
    next.setAttribute('aria-label', 'Next clue');
    next.addEventListener('click', function () { self.engine.step(1); });
    this.headerBody = el('div', 'cg-tools-clue');
    var close = el('button', 'cg-tools-close', '✕');
    close.setAttribute('aria-label', 'Close tools');
    close.addEventListener('click', function () { self.hide(); });
    header.appendChild(prev);
    header.appendChild(this.headerBody);
    header.appendChild(next);
    header.appendChild(close);

    var tabs = el('div', 'cg-tabs');
    tabs.setAttribute('role', 'tablist');
    this.tabButtons = {};
    TABS.forEach(function (tab) {
      var button = el('button', 'cg-tab', tab.label);
      button.setAttribute('role', 'tab');
      // Switching tool by hand drops the picked clue words. They stay
      // highlighted in the clue above otherwise, looking active while the new
      // tool ignores them — and the only way to use them is to unpick and pick
      // again. A highlight has to mean the tool is using that word.
      button.addEventListener('click', function () {
        if (tab.id !== self.tab) self.engine.clearWords();
        self.show(tab.id);
      });
      tabs.appendChild(button);
      self.tabButtons[tab.id] = button;
    });

    this.panel = el('div', 'cg-tool-panel');

    this.container.appendChild(header);
    this.container.appendChild(tabs);
    this.container.appendChild(this.panel);
  };

  ToolsView.prototype.isOpen = function () { return this.open; };

  ToolsView.prototype.show = function (tab) {
    this.open = true;
    this.container.hidden = false;
    this.tab = tab || this.tab;
    var current = this.engine.current();
    this.entryId = current ? current.id : null;
    var self = this;
    Object.keys(this.tabButtons).forEach(function (id) {
      self.tabButtons[id].classList.toggle('is-on', id === self.tab);
      self.tabButtons[id].setAttribute('aria-selected', id === self.tab);
    });
    this.refreshHeader();
    this.renderPanel();
    if (this.options.onOpen) this.options.onOpen(this.tab);
  };

  ToolsView.prototype.hide = function () {
    this.open = false;
    this.container.hidden = true;
    if (this.options.onClose) this.options.onClose();
  };

  ToolsView.prototype.toggle = function (tab) {
    if (this.open && (!tab || tab === this.tab)) this.hide();
    else this.show(tab);
  };

  ToolsView.prototype.refreshHeader = function () {
    var entry = this.engine.current();
    this.headerBody.innerHTML = '';
    if (!entry) return;
    var label = entry.number + ' ' + (entry.dir === ACROSS ? 'Across' : 'Down');
    this.headerBody.appendChild(el('span', 'cg-bar-num', label));
    // Words are clickable here too — this is the clue sitting directly above
    // the anagram box, so it is where changing the fodder is most natural.
    var text = el('span', 'cg-bar-text');
    renderClueWords(this.engine, entry, text);
    this.headerBody.appendChild(text);
    if (entry.enum) {
      this.headerBody.appendChild(el('span', 'cg-bar-enum', '(' + entry.enum + ')'));
    }
    this.headerBody.appendChild(clearEntryButton(this.engine, entry.id));
    if (this.open && this.tab === 'pattern') this.prefillPattern();
  };

  ToolsView.prototype.renderPanel = function () {
    this.panel.innerHTML = '';
    this.message = el('p', 'cg-tool-msg');
    this.message.hidden = true;
    this.panel.appendChild(this.message);
    var builder = {
      anagram: this.buildAnagram,
      pattern: this.buildPattern,
      synonym: this.buildSynonym,
      lookup: this.buildLookup,
      hints: this.buildHints
    }[this.tab];
    if (builder) builder.call(this);

    // Every tool gets the same escape hatch, in the same place, so a wrong
    // turn is one click to undo instead of a field to select and retype.
    var clear = el('button', 'cg-tool-clear', 'Clear');
    clear.type = 'button';
    clear.title = 'Reset this tool';
    var self = this;
    clear.addEventListener('click', function () { self.clearTool(); });
    this.panel.appendChild(clear);
  };

  /* Put the current tool back exactly how it opens. One rule for all five, so
   * "Clear" never needs explaining: the anagram and word tools also drop the
   * clue words they were fed from, or they would simply refill themselves. */
  ToolsView.prototype.clearTool = function () {
    if (this.tab === 'anagram' || this.tab === 'lookup') {
      this.words = [];
      this.engine.clearWords();
    }
    this.renderPanel();
    this.flash('');
  };

  /* `noButton` for a field that searches as you type — a second Search button
   * beside the first would only ask which one to press. */
  ToolsView.prototype._field = function (labelText, value, onRun, noButton) {
    var row = el('div', 'cg-tool-row');
    var label = el('label', 'cg-tool-label', labelText);
    var input = el('input', 'cg-tool-input');
    input.type = 'text';
    input.value = value || '';
    input.spellcheck = false;
    input.autocomplete = 'off';
    var button = noButton ? null : el('button', 'cg-tool-go', 'Search');
    if (button) button.addEventListener('click', function () { onRun(input.value); });
    input.addEventListener('keydown', function (event) {
      // Stop the grid engine swallowing letters typed into a tool field.
      event.stopPropagation();
      if (event.key === 'Enter') onRun(input.value);
    });
    label.appendChild(input);
    row.appendChild(label);
    if (button) row.appendChild(button);
    this.panel.appendChild(row);
    return input;
  };

  ToolsView.prototype._results = function () {
    var box = el('div', 'cg-tool-results');
    this.panel.appendChild(box);
    return box;
  };

  ToolsView.prototype._busy = function (box) {
    box.innerHTML = '';
    box.appendChild(el('p', 'cg-tool-note', 'Searching…'));
  };

  ToolsView.prototype._failed = function (box) {
    box.innerHTML = '';
    box.appendChild(el('p', 'cg-tool-note', 'Could not search just now.'));
  };

  /* A result is clickable: it drops straight into the grid, which is the whole
   * reason for having the tools next to the squares rather than in a tab.
   *
   * `opts.info` puts an ⓘ beside each word — what that word means in a
   * crossword, which is how the site's own match lists work
   * (partials/pattern_results.html:22). Nine words that all fit are nine
   * strings until you can see what they mean.
   *
   * `opts.onMore` makes the "+N more" label a CONTROL rather than a caption.
   * The site's version fetches the rest (partials/helper_results.html:59-63);
   * a label that says more exist and cannot show them is a dead end. */
  ToolsView.prototype._wordList = function (box, words, capped, total, quiet, opts) {
    opts = opts || {};
    box.innerHTML = '';
    if (!words.length) {
      if (!quiet) {
        box.appendChild(el('p', 'cg-tool-note', 'Nothing in the corpus fits that.'));
      }
      return;
    }
    // `quiet` suppresses the count line — the word lookup stacks one list per
    // length and already labels each with its length, so a tally above every
    // one of them is noise.
    var self = this;
    if (!quiet) {
      var count = capped
        ? 'Showing ' + words.length + ' of ' + total
        : words.length + (words.length === 1 ? ' match' : ' matches');
      box.appendChild(el('p', 'cg-tool-note', count));
    } else if (capped) {
      var more = el('span', 'cg-tool-more', '+' + (total - words.length) + ' more');
      if (opts.onMore) {
        more.classList.add('is-live');
        more.setAttribute('role', 'button');
        more.setAttribute('tabindex', '0');
        more.title = 'Show the rest';
        more.addEventListener('click', function () { opts.onMore(); });
        more.addEventListener('keydown', function (event) {
          if (event.key !== 'Enter' && event.key !== ' ') return;
          event.preventDefault();
          opts.onMore();
        });
      }
      box.appendChild(more);
    }
    var list = el('ul', 'cg-word-list');
    // One shared line under the list, as on the site: the ⓘ answers in place
    // rather than opening anything, and a second click on the same word puts
    // it away again.
    var info = opts.info ? el('p', 'cg-tool-info') : null;
    words.forEach(function (word) {
      var item = el('li');
      var button = el('button', 'cg-word-hit', word);
      button.addEventListener('click', function () { self.flash(self.fill(word)); });
      item.appendChild(button);
      if (info) {
        var mark = el('button', 'cg-word-info', 'ⓘ');
        mark.type = 'button';
        mark.title = 'What does ' + word + ' mean?';
        mark.setAttribute('aria-label', 'What does ' + word + ' mean?');
        mark.addEventListener('click', function () { self.showWordInfo(info, word); });
        item.appendChild(mark);
      }
      list.appendChild(item);
    });
    box.appendChild(list);
    if (info) box.appendChild(info);
  };

  /* The ⓘ answer, in the line under the list. Same reverse lookup the site
   * runs (/helper/word-info): what this word means in a crossword. */
  ToolsView.prototype.showWordInfo = function (line, word) {
    if (line.dataset.word === word) {      // second click on the same ⓘ
      line.textContent = '';
      delete line.dataset.word;
      return;
    }
    line.dataset.word = word;
    line.textContent = word + ': looking up…';
    this.api.post('/api/tools/word-info', { word: word }).then(function (data) {
      if (line.dataset.word !== word) return;   // a later ⓘ already won
      var meanings = (data.meanings || []).join(', ');
      line.textContent = word + ': ' + (meanings || 'no definitions recorded');
    }).catch(function () {
      if (line.dataset.word === word) line.textContent = word + ': could not look that up.';
    });
  };

  /* Write a result into the current entry. Returns null on success, or a short
   * reason it could not be placed.
   *
   * A result the same length as the entry IS the answer, so it fills the whole
   * entry wherever the cursor happens to be. A SHORTER result is a component —
   * a synonym or abbreviation that forms part of the answer — so it goes in at
   * the cursor, exactly as typing it would. Anything longer than the entry is
   * never offered as a button in the first place.
   */
  ToolsView.prototype.fill = function (word) {
    var entry = this.engine.current();
    if (!entry) return 'Select a clue first.';
    var letters = word.replace(/[^A-Za-z]/g, '').toUpperCase();
    if (!letters.length) return null;
    if (letters.length > entry.len) {
      return word + ' is too long for this entry.';
    }

    var start = 0;
    if (letters.length < entry.len) {
      start = this.engine.cursor
        ? this.engine.indexInEntry(entry, this.engine.cursor.r, this.engine.cursor.c)
        : 0;
      if (start < 0) start = 0;
      if (start + letters.length > entry.len) {
        // Refuse rather than shunt it somewhere the solver did not choose.
        return word + ' needs ' + letters.length + ' squares — only ' +
          (entry.len - start) + ' left from here.';
      }
    }

    var map = {};
    for (var i = 0; i < letters.length; i++) {
      var cell = entry.cells[start + i];
      map[key(cell[0], cell[1])] = letters[i];
    }
    this.engine.applyLetters(map);

    // Leave the cursor after what was just placed, so components can be
    // dropped in one after another.
    var next = entry.cells[Math.min(start + letters.length, entry.len - 1)];
    this.engine.selectCellKeepingDirection(next[0], next[1]);
    return null;
  };

  ToolsView.prototype.flash = function (message) {
    if (!this.message) return;
    this.message.textContent = message || '';
    this.message.hidden = !message;
  };

  // --- pattern ---------------------------------------------------------

  ToolsView.prototype.buildPattern = function () {
    var self = this;
    this.patternInput = this._field('Pattern', this.currentPattern(), function (value) {
      self.runPattern(value);
    });
    // "Must include" — the site's second pattern field (puzzle.html:368-374).
    // The pattern says WHERE letters go; this says which letters must be in
    // there SOMEWHERE, which is what a solver knows from the wordplay before
    // they know the order. It re-searches as you type, as the live one does.
    this.patternInclude = this._field('Must include', '', function () {
      self.runPattern();
    }, true);
    this.patternInclude.placeholder = 'e.g. LT';
    this.patternInclude.addEventListener('input', function () {
      global.clearTimeout(self._includeTimer);
      self._includeTimer = global.setTimeout(function () { self.runPattern(); }, 400);
    });
    this.patternResults = this._results();
    this.patternResults.appendChild(el('p', 'cg-tool-note',
      'Prefilled from the grid. ? is an unknown square.'));
  };

  ToolsView.prototype.currentPattern = function () {
    var entry = this.engine.current();
    return entry ? this.engine.patternFor(entry.id) : '';
  };

  ToolsView.prototype.prefillPattern = function () {
    if (this.patternInput && document.activeElement !== this.patternInput) {
      this.patternInput.value = this.currentPattern();
    }
  };

  /* The whole journey behind the count, in one call: open the pattern tool on
   * this entry and SEARCH, so the words are on screen without a second action.
   * The live site does exactly this from its "n matches" (puzzle2.js:1215-1220
   * — solverTab('pattern'), fill the box, search). Clicking a result then
   * drops it into the grid, which the results list already does. */
  ToolsView.prototype.openPattern = function () {
    this.show('pattern');
    if (this.patternInput) this.runPattern(this.patternInput.value);
  };

  ToolsView.prototype.runPattern = function (value) {
    var entry = this.engine.current();
    var box = this.patternResults;
    if (value == null) value = this.patternInput ? this.patternInput.value : '';
    if (!value) return;
    this._busy(box);
    var self = this;
    this.api.post('/api/tools/pattern', {
      pattern: value,
      include: this.patternInclude ? this.patternInclude.value : '',
      entry: entry ? entry.id : null
    }).then(function (data) {
      self._wordList(box, data.matches || [], data.capped, data.total,
                     false, { info: true });
    }).catch(function () { self._failed(box); });
  };

  // --- anagram ---------------------------------------------------------

  ToolsView.prototype.buildAnagram = function () {
    var self = this;
    this.anagramInput = this._field('Letters', this.words.join(''), function (value) {
      self.runAnagram(value);
    });
    var toggleRow = el('label', 'cg-tool-check');
    this.anagramUsesGrid = el('input');
    this.anagramUsesGrid.type = 'checkbox';
    this.anagramUsesGrid.checked = true;
    this.anagramUsesGrid.addEventListener('change', function () { self.runAnagram(); });
    toggleRow.appendChild(this.anagramUsesGrid);
    toggleRow.appendChild(el('span', null, 'Filter by the letters already in the grid'));
    this.panel.appendChild(toggleRow);
    this.anagramResults = this._results();
    this.anagramResults.appendChild(el('p', 'cg-tool-note',
      'Click words in the clue above to use them as fodder.'));
  };

  ToolsView.prototype.runAnagram = function (value) {
    if (!this.anagramInput) return;
    if (value == null) {
      value = this.words.length ? this.words.join('') : this.anagramInput.value;
      this.anagramInput.value = value;
    }
    if (!value) return;
    var entry = this.engine.current();
    var box = this.anagramResults;
    this._busy(box);
    var self = this;
    this.api.post('/api/tools/anagram', {
      letters: value,
      entry: entry ? entry.id : null,
      pattern: (this.anagramUsesGrid && this.anagramUsesGrid.checked && entry)
        ? this.engine.patternFor(entry.id) : null
    }).then(function (data) {
      // The site puts the same ⓘ on anagram results as on pattern results
      // (partials/anagram_results.html:21).
      self._wordList(box, data.matches || [], data.capped, data.total,
                     false, { info: true });
    }).catch(function () { self._failed(box); });
  };

  // --- synonym ---------------------------------------------------------

  ToolsView.prototype.buildSynonym = function () {
    var self = this;
    this.synonymInput = this._field('Word or phrase', '', function (value) {
      self.runSynonym(value);
    });
    this.synonymResults = this._results();
    this.synonymResults.appendChild(el('p', 'cg-tool-note',
      'For a word that is not in the clue — a suspected definition, say.'));
  };

  ToolsView.prototype.runSynonym = function (value) {
    if (!value) return;
    var box = this.synonymResults;
    this._busy(box);
    var self = this;
    var entry = this.engine.current();
    this.api.post('/api/tools/synonym', {
      word: value, length: entry ? entry.len : null
    }).then(function (data) {
      box.innerHTML = '';
      if (!(data.synonyms || []).length && !(data.abbreviations || []).length) {
        box.appendChild(el('p', 'cg-tool-note', 'Nothing recorded for that.'));
        return;
      }
      if ((data.synonyms || []).length) {
        box.appendChild(el('h4', 'cg-tool-head', 'Could mean'));
        var wrap = el('div');
        box.appendChild(wrap);
        self._wordList(wrap, data.synonyms, data.capped, data.synonyms.length);
      }
      if ((data.abbreviations || []).length) {
        box.appendChild(el('h4', 'cg-tool-head', 'Abbreviations'));
        box.appendChild(el('p', 'cg-tool-values', data.abbreviations.join(' · ')));
      }
    }).catch(function () { self._failed(box); });
  };

  // --- word lookup -----------------------------------------------------

  ToolsView.prototype.buildLookup = function () {
    this.lookupResults = this._results();
    if (this.words.length) this.runLookup();
    else this.lookupResults.appendChild(el('p', 'cg-tool-note',
      'Click a word in the clue above to look it up.'));
  };

  ToolsView.prototype.runLookup = function () {
    if (!this.lookupResults || !this.words.length) return;
    var box = this.lookupResults;
    this._busy(box);
    var self = this;
    var entry = this.engine.current();
    this.api.post('/api/tools/lookup', {
      // As written, not letters-only: see clueTokens.
      word: this.phrase || this.words.join(' '),
      entry: entry ? entry.id : null
    })
      .then(function (data) {
        box.innerHTML = '';
        box.appendChild(el('h4', 'cg-tool-head', data.word));
        var any = false;

        // Everything that is not LONGER than the entry is clickable. One that
        // matches the length is the answer and fills the entry; a shorter one
        // is a component and goes in at the cursor. Only lengths that cannot
        // physically go in the grid stay as plain text.
        var entryLen = entry ? entry.len : 0;
        (data.meanings || []).forEach(function (group) {
          any = true;
          if (entryLen && group.length <= entryLen) {
            var head = el('p', 'cg-tool-values');
            if (group.fits) head.classList.add('is-fitting');
            head.appendChild(el('span', 'cg-tool-len', group.length + ':'));
            head.appendChild(document.createTextNode(
              group.fits ? ' fits this entry' : ' goes in at the cursor'));
            box.appendChild(head);
            var wrap = el('div');
            box.appendChild(wrap);
            self._wordList(wrap, group.words, !!group.more,
                           group.words.length + group.more, true,
                           { onMore: self._expander(wrap, group.length, false) });
            return;
          }
          var line = el('p', 'cg-tool-values cg-tool-toolong');
          self._lengthLine(line, group.length, group.words, group.more);
          box.appendChild(line);
        });

        if ((data.indicators || []).length) {
          any = true;
          box.appendChild(el('h4', 'cg-tool-head', 'Can indicate'));
          box.appendChild(el('p', 'cg-tool-values', data.indicators.map(function (i) {
            return i.subtype ? i.type + ' (' + i.subtype + ')' : i.type;
          }).join(' · ')));
        }
        if ((data.abbreviations || []).length) {
          any = true;
          box.appendChild(el('h4', 'cg-tool-head', 'Abbreviations'));
          box.appendChild(el('p', 'cg-tool-values', data.abbreviations.join(' · ')));
        }
        if ((data.homophones || []).length) {
          any = true;
          box.appendChild(el('h4', 'cg-tool-head', 'Sounds like'));
          box.appendChild(el('p', 'cg-tool-values', data.homophones.join(' · ')));
        }
        if (!any) {
          box.appendChild(el('p', 'cg-tool-note', 'Nothing recorded for that word.'));
        }
      }).catch(function () { self._failed(box); });
  };

  /* One length's worth of meanings that are too long to go in the grid, drawn
   * as plain text — with the "+N more" still a control. The site expands these
   * the same way it expands the others; being unusable for THIS entry is no
   * reason to hide what the word can mean. */
  ToolsView.prototype._lengthLine = function (line, length, words, more) {
    line.innerHTML = '';
    line.appendChild(el('span', 'cg-tool-len', length + ':'));
    line.appendChild(document.createTextNode(' ' + words.join(' · ')));
    if (!more) return;
    var rest = el('span', 'cg-tool-more is-live', ' +' + more + ' more');
    rest.setAttribute('role', 'button');
    rest.setAttribute('tabindex', '0');
    rest.title = 'Show the rest';
    var run = this._expander(line, length, true);
    rest.addEventListener('click', run);
    rest.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      run();
    });
    line.appendChild(rest);
  };

  /* Fetch every meaning of this clue word at one length and redraw that group
   * in place. The endpoint already takes `letters` — only the browser was
   * never asking, so the label sat there naming a number it would not show. */
  ToolsView.prototype._expander = function (target, length, asText) {
    var self = this;
    return function () {
      var entry = self.engine.current();
      self.api.post('/api/tools/lookup', {
        word: self.phrase || self.words.join(' '),
        entry: entry ? entry.id : null,
        letters: length
      }).then(function (data) {
        var group = (data.meanings || [])[0] || { words: [] };
        if (asText) self._lengthLine(target, length, group.words, 0);
        else self._wordList(target, group.words, false, group.words.length, true);
      }).catch(function () {
        self.flash('Could not fetch the rest just now.');
      });
    };
  };

  // --- hints and explanations ------------------------------------------

  /* The WFW breakdown, drawn the way the site draws it: a clue-type pill, the
   * answer as tiles coloured by the piece that placed each letter, the one-line
   * assembly, then word-by-word rows in clue order. The colours arrive already
   * resolved so this stays a dumb renderer.
   *
   * Row detail may carry `html` — the fodder-letter highlighting, generated by
   * our own renderer with everything else escaped where it is built. It is not
   * user input and not publisher input. */
  function renderBreakdown(breakdown, container) {
    if (!breakdown) return;

    if (breakdown.label) {
      container.appendChild(el('span', 'cg-wfw-type', breakdown.label));
    }

    if ((breakdown.tiles || []).length) {
      var tiles = el('div', 'cg-wfw-tiles');
      breakdown.tiles.forEach(function (tile) {
        if (tile.sep) {
          tiles.appendChild(el('span', 'cg-wfw-gap'));
          return;
        }
        var node = el('span', 'cg-wfw-tile', tile.char);
        if (tile.fg) {
          node.style.color = tile.fg;
          node.style.borderColor = tile.fg;
          node.style.background = tile.fill || 'transparent';
        }
        tiles.appendChild(node);
      });
      container.appendChild(tiles);
    }

    if (breakdown.summary) {
      container.appendChild(el('p', 'cg-wfw-summary', breakdown.summary));
    }

    (breakdown.rows || []).forEach(function (row) {
      var line = el('div', 'cg-wfw-row');
      var pill = el('span', 'cg-wfw-pill', row.pill);
      if (row.fg) {
        pill.style.color = row.fg;
        pill.style.background = row.fill || 'transparent';
      }
      var detail = el('span', 'cg-wfw-detail');
      if (row.html) detail.innerHTML = row.html;
      else detail.textContent = row.text || '';
      line.appendChild(pill);
      line.appendChild(detail);
      container.appendChild(line);
    });
  }

  var HINT_STEPS = [
    { id: 'definition', label: 'Definition' },
    { id: 'clue_type', label: 'Clue type' },
    { id: 'answer', label: 'Answer' },
    { id: 'explanation', label: 'Full explanation' }
  ];

  ToolsView.prototype.buildHints = function () {
    var self = this;
    var entry = this.engine.current();
    this.hintsResults = this._results();
    if (!entry) return;

    var row = el('div', 'cg-hint-row');
    HINT_STEPS.forEach(function (step) {
      var button = el('button', 'cg-hint-step', step.label);
      button.addEventListener('click', function () { self.runHint(step, button); });
      row.appendChild(button);
    });
    this.panel.insertBefore(row, this.hintsResults);

    // Ask which rungs exist WITHOUT fetching their content, so a step that has
    // nothing behind it is never offered and never has to apologise.
    this.api.post('/api/hints/available', { entry: entry.id })
      .then(function (data) {
        var available = data.steps || [];
        HINT_STEPS.forEach(function (step, i) {
          if (available.indexOf(step.id) === -1) {
            row.children[i].disabled = true;
            row.children[i].title = 'Not available for this clue';
          }
        });
        if (!available.length) {
          self.hintsResults.appendChild(el('p', 'cg-tool-note',
            'No explanation has been published for this clue.'));
        }
      }).catch(function () {});
  };

  ToolsView.prototype.runHint = function (step, button) {
    var entry = this.engine.current();
    if (!entry) return;
    var box = this.hintsResults;
    var self = this;
    this.api.post('/api/hints', { entry: entry.id, step: step.id })
      .then(function (data) {
        button.classList.add('is-used');
        var block = el('div', 'cg-hint-block');
        block.appendChild(el('h4', 'cg-tool-head', step.label));
        if (data.value == null) {
          block.appendChild(el('p', 'cg-tool-note',
            data.unavailable || 'Not available for this clue.'));
        } else if (step.id === 'explanation') {
          renderBreakdown(data.value, block);
        } else {
          block.appendChild(el('p', 'cg-hint-value', data.value));
        }
        box.appendChild(block);
      }).catch(function () { self._failed(box); });
  };

  /* ------------------------------------------------------------------ */

  global.Cordelia = {
    Engine: Engine,
    Api: Api,
    GridView: GridView,
    ClueListView: ClueListView,
    ClueBarView: ClueBarView,
    MatchCount: MatchCount,
    ToolsView: ToolsView,
    TABS: TABS,
    ACROSS: ACROSS,
    DOWN: DOWN,
    key: key,
    el: el
  };
}(window));
