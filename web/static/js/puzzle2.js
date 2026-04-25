/**
 * Puzzle page JavaScript — solve mode, grid, crossings, word help.
 *
 * Requires window.PUZZLE_CONFIG to be set before this script loads:
 *   { source, puzzleType, puzzleNumber, isPrize }
 */

/* --- Config --- */
var _cfg = window.PUZZLE_CONFIG || {};
var _solveKey = 'solve_' + _cfg.source + '_' + _cfg.puzzleNumber;

/* --- Tools overlay --- */
var _toolsClueId = null;
var _toolsScrollY = 0;

function openToolsOverlay(clueId) {
    _clearSelection();
    _toolsClueId = clueId;
    _toolsScrollY = window.scrollY;
    _lastFocusedClueId = clueId;

    var input = document.querySelector('.solve-answer[data-clue-id="' + clueId + '"]');
    if (!input) return;

    var overlay = document.getElementById('tools-overlay');
    var label = input.dataset.label || '';
    var clueText = input.dataset.clueText || '';
    var enumVal = input.dataset.enum || '';

    document.getElementById('tools-overlay-label').textContent = label;
    _buildOverlayClueWords(document.getElementById('tools-overlay-clue'), clueText);
    document.getElementById('tools-overlay-enum').textContent = enumVal ? '(' + enumVal + ')' : '';

    // Sync answer from clue list into overlay
    var overlayAnswer = document.getElementById('tools-overlay-answer');
    overlayAnswer.value = input.value;
    overlayAnswer.placeholder = enumVal || 'answer';
    document.getElementById('tools-overlay-result').textContent = '';

    // Set up pattern with crossing if available
    var patInput = document.getElementById('pattern-input');
    var enumInput = document.getElementById('pattern-enum');
    if (enumInput) enumInput.value = enumVal;
    if (typeof _updatePatternEnumToggle === 'function') _updatePatternEnumToggle(enumVal);
    var crossing = input.getAttribute('data-crossing');
    if (crossing && patInput) {
        patInput.value = crossing.replace(/_/g, '?');
    } else if (patInput) {
        // Set up blank pattern from enumeration
        var nums = enumVal.match(/\d+/g);
        if (nums) {
            var parts = nums.map(function(n) { return '?'.repeat(parseInt(n)); });
            patInput.value = parts.join('-');
        } else {
            patInput.value = '';
        }
    }

    // Pre-fetch similar clues
    if (clueText) {
        var simUrl = '/helper/similar?q=' + encodeURIComponent(clueText) + '&ht=' + _ht;
        if (enumVal) simUrl += '&enum=' + encodeURIComponent(enumVal);
        if (label) simUrl += '&label=' + encodeURIComponent(label);
        if (clueId) simUrl += '&clue_id=' + encodeURIComponent(clueId);
        fetch(simUrl).then(function(r) { return r.text(); }).then(function(html) {
            input.setAttribute('data-similar-html', html);
        });
    }

    // Default to anagram tab, clear stale results and tab cache
    document.getElementById('solver-results').innerHTML = '';
    _tabResultsCache = {};
    solverTab('anagram');
    overlay.classList.remove('hidden');
}

function closeToolsOverlay() {
    _clearSelection();
    var overlay = document.getElementById('tools-overlay');
    overlay.classList.add('hidden');
    document.getElementById('solver-results').innerHTML = '';
    _tabResultsCache = {};
    // Clear anagram chips so fodder doesn't carry to next clue
    if (typeof anagramClear === 'function') anagramClear();
    // Restore scroll position
    window.scrollTo(0, _toolsScrollY);
    _toolsClueId = null;
}

function _buildOverlayClueWords(container, text) {
    container.innerHTML = '';
    var parts = text.split(/(\s+)/);
    var wordIdx = 0;
    parts.forEach(function(part) {
        if (part.trim()) {
            var clean = part.replace(/[^A-Za-z]/g, '').toLowerCase();
            if (clean) {
                var span = document.createElement('span');
                span.className = 'clue-word cursor-pointer hover:bg-indigo-100 hover:rounded px-0.5 -mx-0.5 transition-colors';
                span.dataset.idx = wordIdx;
                span.dataset.clean = clean;
                span.textContent = part;
                span.onclick = function() { overlayWordHelp(this); };
                container.appendChild(span);
                wordIdx++;
            } else {
                container.appendChild(document.createTextNode(part));
            }
        } else {
            container.appendChild(document.createTextNode(part));
        }
    });
}

function overlayWordHelp(span) {
    var idx = parseInt(span.dataset.idx);
    var clean = span.dataset.clean;

    // If anagram tab is open, send word there directly
    if (_isAnagramMode()) {
        anagramFromWord(clean);
        return;
    }

    // If this word is already selected and it's the only one, toggle off
    if (_selectedWords.length === 1 && _selectedWords[0].idx === idx) {
        _clearSelection();
        document.getElementById('solver-results').innerHTML = '';
        return;
    }

    // If adjacent to current selection, extend it
    if (_selectedWords.length > 0) {
        var minIdx = Math.min.apply(null, _selectedWords.map(function(w) { return w.idx; }));
        var maxIdx = Math.max.apply(null, _selectedWords.map(function(w) { return w.idx; }));
        if (idx === minIdx - 1 || idx === maxIdx + 1) {
            _selectedWords.push({el: span, idx: idx, clean: clean, clueId: _toolsClueId});
            span.classList.add('bg-indigo-200', 'rounded');
        } else {
            _clearSelection();
            _selectedWords.push({el: span, idx: idx, clean: clean, clueId: _toolsClueId});
            span.classList.add('bg-indigo-200', 'rounded');
        }
    } else {
        _selectedWords.push({el: span, idx: idx, clean: clean, clueId: _toolsClueId});
        span.classList.add('bg-indigo-200', 'rounded');
    }

    // Build the lookup phrase from selected words in order
    _selectedWords.sort(function(a, b) { return a.idx - b.idx; });
    var phrase = _selectedWords.map(function(w) { return w.clean; }).join(' ');

    // Word lookup in the overlay results area — include enum so filter button appears
    var lookupUrl = '/helper/lookup?word=' + encodeURIComponent(phrase) + '&ht=' + _ht;
    if (_toolsClueId) {
        var _luAns = document.querySelector('.solve-answer[data-clue-id="' + _toolsClueId + '"]');
        if (_luAns && _luAns.dataset.enum) lookupUrl += '&enum=' + encodeURIComponent(_luAns.dataset.enum);
    }
    htmx.ajax('GET', lookupUrl, {target: '#solver-results', swap: 'innerHTML'});
}

function _toolsFillAnswer(word) {
    // Fill the answer in both the overlay and the clue list input, then close
    if (_toolsClueId) {
        var input = document.querySelector('.solve-answer[data-clue-id="' + _toolsClueId + '"]');
        if (input) {
            input.value = word;
            _saveSolveAnswer(_toolsClueId, word);
        }
        var overlayAnswer = document.getElementById('tools-overlay-answer');
        if (overlayAnswer) overlayAnswer.value = word;
    }
    closeToolsOverlay();
}

function toolsOverlayCheck() {
    var overlayAnswer = document.getElementById('tools-overlay-answer');
    var guess = overlayAnswer.value.replace(/\s/g, '').toUpperCase();
    if (!guess || !_toolsClueId) return;

    // Sync to clue list
    var input = document.querySelector('.solve-answer[data-clue-id="' + _toolsClueId + '"]');
    if (input) input.value = overlayAnswer.value;

    // Run normal check
    if (input) solveCheck(input);

    // Show result in overlay too
    var card = input ? input.closest('.clue-card') : null;
    var answer = card ? (card.dataset.answer || '').replace(/\s/g, '').toUpperCase() : '';
    var result = document.getElementById('tools-overlay-result');
    if (!answer) {
        result.className = 'text-xs text-amber-600';
        result.textContent = 'Saved';
    } else if (guess === answer) {
        result.className = 'text-xs text-green-600 font-bold';
        result.textContent = 'Correct!';
    } else {
        result.className = 'text-xs text-red-500';
        result.textContent = 'Not right';
    }
}

function toolsOverlayAddToGrid() {
    var overlayAnswer = document.getElementById('tools-overlay-answer');
    if (!overlayAnswer.value || !_toolsClueId) return;

    // Sync to clue list input and add to grid
    var input = document.querySelector('.solve-answer[data-clue-id="' + _toolsClueId + '"]');
    if (input) {
        input.value = overlayAnswer.value;
        solveAddToGrid(input);
    }
    closeToolsOverlay();
}

/* --- Grid toggle --- */
function toggleGrid(btn, url) {
    var area = document.getElementById('grid-area');
    var showLabel = _solveMode ? 'Show grid' : 'Show full grid';
    if (area.innerHTML.trim()) {
        area.innerHTML = '';
        btn.textContent = showLabel;
    } else {
        btn.textContent = 'Loading...';
        htmx.ajax('GET', url, {target: '#grid-area', swap: 'innerHTML'}).then(function() {
            btn.textContent = 'Hide grid';
        });
    }
}

/* --- Hint toggle --- */
function hideHints(id) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = '';
    var btn = el.parentElement.querySelector('.hint-hide');
    if (btn) btn.classList.add('hidden');
}

/* --- Panel management --- */
function closeAllPanels(except) {
    // Close word helpers
    if (except !== 'wordhelp') {
        document.querySelectorAll('[id^="wordhelp-"]').forEach(function(e) { e.innerHTML = ''; });
        _clearSelection();
    }
    // Close hint reveals
    if (except !== 'hints') {
        hideAllHints();
    }
    // Close solver panel (only exists on non-overlay pages)
    if (except !== 'solver') {
        var panel = document.getElementById('helper-panel');
        if (panel) panel.classList.add('hidden');
        var openBtn = document.getElementById('helper-open');
        if (openBtn) openBtn.classList.remove('hidden');
    }
    // Close grid (but never in solve mode — grid persists)
    if (except !== 'grid' && !_solveMode) {
        var gridArea = document.getElementById('grid-area');
        if (gridArea) gridArea.innerHTML = '';
        var gridBtn = document.getElementById('grid-toggle');
        if (gridBtn) gridBtn.textContent = 'Show full grid';
    }
}

/* --- Word help --- */
var _selectedWords = []; // [{el, idx, clean, clueId}]
function _isAnagramMode() {
    var overlay = document.getElementById('tools-overlay');
    var panel = document.getElementById('helper-panel');
    var tab = document.getElementById('solver-anagram');
    var containerOpen = (overlay && !overlay.classList.contains('hidden')) ||
                        (panel && !panel.classList.contains('hidden'));
    return containerOpen && tab && !tab.classList.contains('hidden');
}
function wordHelp(span) {
    var idx = parseInt(span.dataset.idx);
    var clean = span.dataset.clean;
    var clueId = span.dataset.clue;
    var target = 'wordhelp-' + clueId;
    _lastFocusedClueId = clueId;

    // Close other panels when opening word help
    closeAllPanels('wordhelp');

    // If anagram tab is open, send word there instead
    if (_isAnagramMode()) {
        _lastFocusedClueId = clueId;
        anagramFromWord(clean);
        return;
    }

    // Close solver panel when word help opens (only exists on non-overlay pages)
    var panel = document.getElementById('helper-panel');
    if (panel && !panel.classList.contains('hidden')) {
        panel.classList.add('hidden');
        var openBtn = document.getElementById('helper-open');
        if (openBtn) openBtn.classList.remove('hidden');
    }

    // If clicking in a different clue, start fresh
    if (_selectedWords.length > 0 && _selectedWords[0].clueId !== clueId) {
        _clearSelection();
    }

    // If this word is already selected and it's the only one, toggle off
    if (_selectedWords.length === 1 && _selectedWords[0].idx === idx) {
        _clearSelection();
        document.getElementById(target).innerHTML = '';
        return;
    }

    // If adjacent to current selection, extend it
    if (_selectedWords.length > 0) {
        var minIdx = Math.min.apply(null, _selectedWords.map(function(w) { return w.idx; }));
        var maxIdx = Math.max.apply(null, _selectedWords.map(function(w) { return w.idx; }));
        if (idx === minIdx - 1 || idx === maxIdx + 1) {
            // Extend selection
            _selectedWords.push({el: span, idx: idx, clean: clean, clueId: clueId});
            span.classList.add('bg-indigo-200', 'rounded');
        } else {
            // Non-adjacent — start fresh
            _clearSelection();
            _selectedWords.push({el: span, idx: idx, clean: clean, clueId: clueId});
            span.classList.add('bg-indigo-200', 'rounded');
        }
    } else {
        _selectedWords.push({el: span, idx: idx, clean: clean, clueId: clueId});
        span.classList.add('bg-indigo-200', 'rounded');
    }

    // Build the lookup phrase from selected words in order
    _selectedWords.sort(function(a, b) { return a.idx - b.idx; });
    var phrase = _selectedWords.map(function(w) { return w.clean; }).join(' ');

    // Show inline below the clue
    document.querySelectorAll('[id^="wordhelp-"]').forEach(function(e) {
        if (e.id !== target) e.innerHTML = '';
    });
    // Pass enumeration for optional filtering
    var enumVal = '';
    var solveInput = document.querySelector('.solve-answer[data-clue-id="' + clueId + '"]');
    if (solveInput) enumVal = solveInput.dataset.enum || '';
    var url = '/helper/lookup?word=' + encodeURIComponent(phrase) + '&ht=' + _ht;
    if (enumVal) url += '&enum=' + encodeURIComponent(enumVal);
    htmx.ajax('GET', url, {target: '#' + target, swap: 'innerHTML'});
}
function _clearSelection() {
    _selectedWords.forEach(function(w) {
        w.el.classList.remove('bg-indigo-200', 'rounded');
    });
    _selectedWords = [];
}
function hideAllHints() {
    document.querySelectorAll('[id^="hints-"]').forEach(function(el) {
        el.innerHTML = '';
    });
    document.querySelectorAll('.hint-hide').forEach(function(btn) {
        btn.classList.add('hidden');
    });
}
document.body.addEventListener('htmx:beforeRequest', function(e) {
    if (e.target.id && e.target.id.startsWith('hints-')) {
        closeAllPanels('hints');
    }
});
document.body.addEventListener('htmx:afterSwap', function(e) {
    if (e.target.id && e.target.id.startsWith('hints-')) {
        var btn = e.target.parentElement.querySelector('.hint-hide');
        if (btn && e.target.innerHTML.trim()) btn.classList.remove('hidden');
    }
    // When word lookup results appear with grouped meanings, show tip about clicking the number
    if (e.target.id && e.target.id.startsWith('wordhelp-')) {
        if (e.target.querySelector('[id^="meanings-"]')) {
            CordeliaTips.init([
                {id: 'wordhelp-expand', text: 'See those numbers in brackets? Click one to see all the matches for that word length. Really useful when you know how many letters you need.'}
            ]);
        }
    }
});

/* --- Solve Mode --- */
var _solveMode = false;

function toggleSolveMode() {
    // Block solve mode if no grid is available
    if (!_solveMode && !_cfg.hasGrid) {
        var btn = document.getElementById('solve-toggle');
        var orig = btn.textContent;
        btn.textContent = 'No grid available for this puzzle';
        btn.classList.add('bg-red-100', 'text-red-700', 'border-red-300');
        setTimeout(function() {
            btn.textContent = orig;
            btn.classList.remove('bg-red-100', 'text-red-700', 'border-red-300');
        }, 3000);
        return;
    }
    _solveMode = !_solveMode;
    var btn = document.getElementById('solve-toggle');
    if (_solveMode) {
        btn.textContent = 'Exit solve';
        btn.classList.remove('border-emerald-400', 'text-emerald-700', 'hover:bg-emerald-50');
        btn.classList.add('bg-emerald-600', 'text-white', 'hover:bg-emerald-700');
        _enterSolveMode();
    } else {
        btn.textContent = 'Solve';
        btn.classList.add('border-emerald-400', 'text-emerald-700', 'hover:bg-emerald-50');
        btn.classList.remove('bg-emerald-600', 'text-white', 'hover:bg-emerald-700');
        _exitSolveMode();
    }
    localStorage.setItem(_solveKey + '_active', _solveMode ? '1' : '');
}

function _prefillDbAnswers() {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    var changed = false;
    document.querySelectorAll('.clue-card').forEach(function(card) {
        var clueId = card.dataset.clueId;
        var answer = (card.dataset.answer || '').replace(/\s/g, '').toUpperCase();
        if (answer && !state[clueId]) {
            state[clueId] = { value: answer, correct: true };
            changed = true;
        }
    });
    if (changed) {
        localStorage.setItem(_solveKey, JSON.stringify(state));
    }
}

function _enterSolveMode() {
    // Hide all hint buttons and badges
    document.querySelectorAll('[id^="explain-"]').forEach(function(el) {
        el.classList.add('solve-hidden');
        el.style.display = 'none';
    });
    // Hide source/tier badges in solve mode
    document.querySelectorAll('.solve-hide-badges').forEach(function(el) {
        el.style.display = 'none';
    });
    // Hide any revealed hints
    document.querySelectorAll('[id^="hints-"]').forEach(function(el) {
        el.innerHTML = '';
    });
    // Show solve inputs, hide "See X" ref clues
    document.querySelectorAll('.solve-input').forEach(function(el) {
        el.classList.remove('hidden');
    });
    document.querySelectorAll('.linked-ref').forEach(function(el) {
        el.classList.add('hidden');
    });
    // Show progress and grid button
    document.getElementById('solve-progress').classList.remove('hidden');
    document.getElementById('solve-grid-btn').classList.remove('hidden');
    var saveBtn = document.getElementById('solve-save-btn');
    if (saveBtn) saveBtn.classList.remove('hidden');
    // Admin: pre-populate solve state with existing DB answers so crossings
    // appear for unsolved clues and the grid shows all known letters.
    if (_cfg.isAdmin) {
        _prefillDbAnswers();
    }
    // Restore saved answers and crossings
    _restoreSolveState();
    _updateProgress();
    _restoreCachedCrossings() || _fetchCrossings();
    // Auto-show grid — it persists throughout solve mode
    showSolveGrid();
}

function _exitSolveMode() {
    // Show hint buttons and badges again
    document.querySelectorAll('[id^="explain-"]').forEach(function(el) {
        el.classList.remove('solve-hidden');
        el.style.display = '';
    });
    document.querySelectorAll('.solve-hide-badges').forEach(function(el) {
        el.style.display = '';
    });
    // Hide solve inputs, show "See X" ref clues again
    document.querySelectorAll('.solve-input').forEach(function(el) {
        el.classList.add('hidden');
    });
    document.querySelectorAll('.linked-ref').forEach(function(el) {
        el.classList.remove('hidden');
    });
    // Hide progress and grid button
    document.getElementById('solve-progress').classList.add('hidden');
    document.getElementById('solve-grid-btn').classList.add('hidden');
    var saveBtn = document.getElementById('solve-save-btn');
    if (saveBtn) saveBtn.classList.add('hidden');
    // Clear the solve-mode grid and reset button text
    var gridArea = document.getElementById('grid-area');
    if (gridArea) gridArea.innerHTML = '';
    document.getElementById('grid-toggle').textContent = 'Show full grid';
}

function toggleSolveHints(clueId) {
    var row = document.getElementById('explain-' + clueId);
    if (!row) return;
    var isHidden = row.style.display === 'none';
    if (isHidden) {
        row.style.display = '';
    } else {
        row.style.display = 'none';
        // Clear any revealed hint content when hiding
        var hints = document.getElementById('hints-' + clueId);
        if (hints) hints.innerHTML = '';
    }
}

function solveCheckOrDelete(btn) {
    var input = btn.closest('.solve-input').querySelector('.solve-answer');
    if (btn.getAttribute('data-mode') === 'delete') {
        solveDelete(input, btn);
    } else {
        solveCheck(input);
    }
}

function solveDelete(input, btn) {
    var clueId = input.dataset.clueId;
    var card = input.closest('.clue-card');
    var linkedId = card && card.dataset.linkedId;
    // Remove from localStorage
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    delete state[clueId];
    if (linkedId) delete state[linkedId];
    localStorage.setItem(_solveKey, JSON.stringify(state));
    // Reset input
    input.value = '';
    input.disabled = false;
    input.classList.remove('border-green-400', 'bg-green-50', 'border-indigo-400', 'bg-indigo-50', 'border-red-400');
    input.classList.add('border-gray-300');
    // Reset button back to Check
    btn.textContent = 'Check';
    btn.removeAttribute('data-mode');
    btn.classList.remove('bg-red-500', 'hover:bg-red-600');
    btn.classList.add('bg-emerald-600', 'hover:bg-emerald-700');
    // Clear result
    var result = input.parentElement.querySelector('.solve-result');
    result.className = 'solve-result text-xs';
    result.textContent = '';
    // Show crossing pattern again (was hidden on solve/add)
    var crossEl = input.parentElement.querySelector('.solve-crossing');
    if (crossEl) crossEl.classList.remove('hidden');
    _updateProgress();
    // Recalculate crossings without the deleted answer
    localStorage.removeItem(_crossingsCacheKey);
    _fetchCrossings();
    // Refresh the persistent grid without the deleted answer
    showSolveGrid();
}

function _makeDeleteBtn(btn) {
    btn.textContent = 'Delete';
    btn.setAttribute('data-mode', 'delete');
    btn.classList.remove('bg-emerald-600', 'hover:bg-emerald-700');
    btn.classList.add('bg-red-500', 'hover:bg-red-600');
}

function solveCheck(input) {
    var clueId = input.dataset.clueId;
    var card = input.closest('.clue-card');
    var answer = (card.dataset.answer || '').replace(/\s/g, '').toUpperCase();
    var guess = input.value.replace(/\s/g, '').toUpperCase();
    var result = input.parentElement.querySelector('.solve-result');

    if (!guess) return;

    // Length check
    var expectedLen = _getExpectedLength(input.dataset.enum || '');
    if (expectedLen > 0) {
        if (guess.length < expectedLen) {
            var short = expectedLen - guess.length;
            result.className = 'solve-result text-xs text-amber-600';
            result.textContent = 'Still needs ' + short + ' more letter' + (short === 1 ? '' : 's');
            _saveSolveAnswer(clueId, input.value);
            return;
        }
        if (guess.length > expectedLen) {
            result.className = 'solve-result text-xs text-red-500';
            result.textContent = 'This clue needs ' + expectedLen + ' letters \u2014 you\u2019ve entered ' + guess.length;
            _saveSolveAnswer(clueId, input.value);
            return;
        }
    }

    // Save to localStorage
    _saveSolveAnswer(clueId, input.value);

    if (!answer) {
        result.className = 'solve-result text-xs text-amber-600';
        result.textContent = 'No check available';
        _updateProgress();
        return;
    }

    if (guess === answer) {
        result.className = 'solve-result text-xs text-green-600 font-bold';
        result.textContent = 'Correct!';
        input.classList.remove('border-gray-300', 'border-red-400');
        input.classList.add('border-green-400', 'bg-green-50');
        input.disabled = true;
        _saveSolveAnswer(clueId, input.value, true);
        _makeDeleteBtn(input.parentElement.querySelector('.solve-check-btn'));
        var crossEl = input.parentElement.querySelector('.solve-crossing');
        if (crossEl) crossEl.classList.add('hidden');
        // Show hint buttons for this solved clue so user can view explanation
        var explainEl = document.getElementById('explain-' + clueId);
        if (explainEl) { explainEl.classList.remove('solve-hidden'); explainEl.style.display = ''; }
        _fetchCrossings();
        // Cordelia tip: add to grid
        CordeliaTips.init([
            {id: 'solve-correct-grid', text: 'Got it! Now hit <strong>Add to grid</strong> to place it. The grid at the top updates automatically &mdash; use <strong>Go to grid</strong> to scroll up and see your progress.'}
        ]);
    } else {
        var conflict = _checkCrossingConflict(input, guess);
        if (conflict) {
            result.className = 'solve-result text-xs text-amber-600';
            result.textContent = 'The letter at position ' + conflict.position + ' must be ' + conflict.expected + ' \u2014 does your answer fit?';
        } else {
            result.className = 'solve-result text-xs text-red-500';
            result.innerHTML = 'Not right &nbsp;<span class="text-gray-400 cursor-pointer hover:text-red-500 underline" onclick="this.closest(\'.solve-input\').querySelector(\'.solve-answer\').value=\'\';this.closest(\'.solve-input\').querySelector(\'.solve-answer\').classList.remove(\'border-red-400\');this.closest(\'.solve-input\').querySelector(\'.solve-answer\').classList.add(\'border-gray-300\');this.closest(\'.solve-input\').querySelector(\'.solve-answer\').disabled=false;this.closest(\'.solve-input\').querySelector(\'.solve-answer\').focus();this.parentElement.textContent=\'\'">Clear</span>';
            input.classList.remove('border-gray-300', 'border-green-400');
            input.classList.add('border-red-400');
            // Shake animation
            input.style.animation = 'none';
            input.offsetHeight; // trigger reflow
            input.style.animation = 'shake 0.3s';
        }
    }
    _updateProgress();
}

function solveAutoCheck(input) {
    // Auto-check on blur: only if there's text and no result shown yet
    var guess = input.value.replace(/\s/g, '').toUpperCase();
    if (!guess) return;
    if (input.disabled) return;  // already solved

    var result = input.parentElement.querySelector('.solve-result');
    if (result && result.textContent.trim()) return;  // already has feedback

    var card = input.closest('.clue-card');
    var answer = (card.dataset.answer || '').replace(/\s/g, '').toUpperCase();

    if (!answer) {
        _saveSolveAnswer(input.dataset.clueId, input.value);
        result.className = 'solve-result text-xs text-amber-600';
        result.textContent = 'No check available';
        return;
    }

    // Check length — parse enumeration to get expected letter count
    var enumStr = input.dataset.enum || '';
    var expectedLen = 0;
    var nums = enumStr.match(/\d+/g);
    if (nums) expectedLen = nums.reduce(function(a, b) { return a + parseInt(b); }, 0);

    if (expectedLen > 0 && guess.length < expectedLen) {
        // Partial answer — just save, no feedback
        _saveSolveAnswer(input.dataset.clueId, input.value);
        return;
    }

    // Full-length answer — nudge user to click Check (don't auto-solve,
    // as the blur+click race converts the button to Delete before the
    // click event fires, causing the answer to vanish)
    _saveSolveAnswer(input.dataset.clueId, input.value);
    result.className = 'solve-result text-xs text-amber-500';
    result.textContent = 'Try Check \u2192';
}

function _saveSolveAnswer(clueId, value, correct) {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    state[clueId] = {value: value, correct: !!correct};
    localStorage.setItem(_solveKey, JSON.stringify(state));
}

function _restoreSolveState() {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    document.querySelectorAll('.solve-answer').forEach(function(input) {
        var clueId = input.dataset.clueId;
        var saved = state[clueId];
        if (saved) {
            input.value = saved.value;
            if (saved.correct) {
                input.classList.add('border-green-400', 'bg-green-50');
                input.disabled = true;
                var result = input.parentElement.querySelector('.solve-result');
                result.className = 'solve-result text-xs text-green-600 font-bold';
                result.textContent = 'Correct!';
                _makeDeleteBtn(input.parentElement.querySelector('.solve-check-btn'));
            }
        }
    });
}

function _updateProgress() {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    var total = document.querySelectorAll('.clue-card').length;
    var solved = 0;
    for (var k in state) {
        if (state[k].correct) solved++;
    }
    var el = document.getElementById('solve-progress');
    el.textContent = solved + '/' + total + ' solved';
    _trackActiveSolve(solved, total);
}

var _activeSolvesKey = 'cordelia_active_solves';

function _trackActiveSolve(solved, total) {
    if (!_solveMode) return;
    var puzzleId = _cfg.source + '/' + _cfg.puzzleType + '/' + _cfg.puzzleNumber;
    var url = '/' + puzzleId;
    var solves = [];
    try { solves = JSON.parse(localStorage.getItem(_activeSolvesKey) || '[]'); } catch(e) { solves = []; }

    // Remove any existing entry for this puzzle
    solves = solves.filter(function(s) { return s.id !== puzzleId; });

    // Only track if there's progress but puzzle isn't complete
    if (solved > 0 && solved < total) {
        solves.unshift({
            id: puzzleId,
            source: _cfg.sourceName || _cfg.source,
            typeLabel: _cfg.typeLabel || _cfg.puzzleType,
            number: _cfg.puzzleNumber,
            date: _cfg.publicationDate || '',
            progress: solved + '/' + total,
            url: url,
            lastActive: new Date().toISOString()
        });
    }

    // Keep only the 10 most recent
    solves = solves.slice(0, 10);
    localStorage.setItem(_activeSolvesKey, JSON.stringify(solves));
}

function _removeActiveSolve() {
    var puzzleId = _cfg.source + '/' + _cfg.puzzleType + '/' + _cfg.puzzleNumber;
    var solves = [];
    try { solves = JSON.parse(localStorage.getItem(_activeSolvesKey) || '[]'); } catch(e) { solves = []; }
    solves = solves.filter(function(s) { return s.id !== puzzleId; });
    localStorage.setItem(_activeSolvesKey, JSON.stringify(solves));
}

function scrollToClue(num) {
    var across = document.getElementById('clue-' + num + '-across');
    var down = document.getElementById('clue-' + num + '-down');
    // Go to across first (or down if no across)
    var el = across || down;
    _scrollToClueEl(el, num, across && down ? (el === across ? 'down' : 'across') : null);
}
function _scrollToClueEl(el, num, otherDir) {
    if (!el) return;
    // Remove any existing "also see" links
    document.querySelectorAll('.also-see-link').forEach(function(e) { e.remove(); });
    var y = el.getBoundingClientRect().top + window.pageYOffset - 80;
    window.scrollTo({top: y, behavior: 'instant'});
    el.classList.add('bg-indigo-50');
    setTimeout(function() { el.classList.remove('bg-indigo-50'); }, 3000);
    // Show "Also see" link if both directions exist
    if (otherDir) {
        var link = document.createElement('span');
        link.className = 'also-see-link text-xs text-indigo-500 hover:text-indigo-700 cursor-pointer underline ml-2';
        link.textContent = '\u2192 ' + num + otherDir[0];
        link.onclick = function() {
            link.remove();
            var other = document.getElementById('clue-' + num + '-' + otherDir);
            _scrollToClueEl(other, num, null);
        };
        el.querySelector('.flex-1') && el.querySelector('.flex-1').appendChild(link);
        setTimeout(function() { if (link.parentNode) link.remove(); }, 5000);
    }
    if (_solveMode) {
        // Close everything and just scroll — don't auto-focus input
        // (auto-focus opens keyboard + solver panel on mobile which is chaos)
        closeAllPanels();
        CordeliaTips._remove();
    }
}

function solveAddToGrid(input) {
    var guess = input.value.replace(/\s/g, '').toUpperCase();
    if (!guess) return;
    var result = input.parentElement.querySelector('.solve-result');

    // Length check
    var expectedLen = _getExpectedLength(input.dataset.enum || '');
    if (expectedLen > 0) {
        if (guess.length < expectedLen) {
            var short = expectedLen - guess.length;
            result.className = 'solve-result text-xs text-amber-600';
            result.textContent = 'Still needs ' + short + ' more letter' + (short === 1 ? '' : 's');
            return;
        }
        if (guess.length !== expectedLen) {
            result.className = 'solve-result text-xs text-red-500';
            result.textContent = 'This clue needs ' + expectedLen + ' letters \u2014 you\u2019ve entered ' + guess.length;
            return;
        }
    }

    // Crossing conflict check — block on Add to grid
    var conflict = _checkCrossingConflict(input, guess);
    if (conflict) {
        result.className = 'solve-result text-xs text-red-500';
        result.textContent = 'The letter at position ' + conflict.position + ' must be ' + conflict.expected + ' \u2014 does your answer fit?';
        return;
    }

    var clueId = input.dataset.clueId;
    _saveSolveAnswer(clueId, input.value, true);
    // Also save for linked clue if this is a spanning clue
    var card = input.closest('.clue-card');
    var linkedId = card && card.dataset.linkedId;
    if (linkedId) {
        _saveSolveAnswer(linkedId, input.value, true);
    }
    input.classList.add('border-indigo-400', 'bg-indigo-50');
    var result = input.parentElement.querySelector('.solve-result');
    result.className = 'solve-result text-xs text-indigo-600';
    result.textContent = 'Added to grid';
    _makeDeleteBtn(input.parentElement.querySelector('.solve-check-btn'));
    var crossEl = input.parentElement.querySelector('.solve-crossing');
    if (crossEl) crossEl.classList.add('hidden');
    _updateProgress();
    _fetchCrossings();
    closeAllPanels();
    // Refresh the persistent grid with the newly added answer
    showSolveGrid();
    // Queue enrichment from explanation pieces
    fetch('/admin/queue-enrichment/' + clueId, {method: 'POST'}).catch(function(){});
}

function _getSolvedIds() {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    var ids = [];
    for (var k in state) {
        if (state[k].correct) ids.push(k);
    }
    return ids;
}

var _revealedCells = {};  // kept empty — reveals are now fleeting

function revealGridLetter(td) {
    var letter = td.dataset.solution;
    if (!letter) return;
    if (td._peekTimer) return;  // already peeking
    // Flash the letter briefly
    var span = td.querySelector('span:last-child');
    span.textContent = letter;
    span.classList.add('text-indigo-600');
    td.classList.add('bg-indigo-50');
    td.classList.remove('hover:bg-indigo-50');
    // Hide after 1.5 seconds
    td._peekTimer = setTimeout(function() {
        span.textContent = '';
        span.classList.remove('text-indigo-600');
        td.classList.remove('bg-indigo-50');
        td.classList.add('hover:bg-indigo-50');
        td._peekTimer = null;
    }, 1500);
}

function _puzzleUrl(path) {
    return '/' + _cfg.source + '/' + _cfg.puzzleType + '/' + _cfg.puzzleNumber + '/' + path;
}

function showSolveGrid() {
    _cancelMatchCountFetches();
    closeAllPanels('grid');
    var solvedIds = _getSolvedIds();
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    // Build user answers for clues where DB might not have the answer
    var userAnswers = {};
    for (var k in state) {
        if (state[k].correct && state[k].value) {
            userAnswers[k] = state[k].value.replace(/\s/g, '').toUpperCase();
        }
    }
    var url = _puzzleUrl('grid-progress') + '?solved=' + solvedIds.join(',')
            + '&answers=' + encodeURIComponent(JSON.stringify(userAnswers));
    var gridArea = document.getElementById('grid-area');
    htmx.ajax('GET', url, {target: '#grid-area', swap: 'innerHTML'}).then(function() {
        // Don't fetch crossings here — they're already on the page.
        // Crossings are only re-fetched after Add to Grid when they change.
        CordeliaTips.init([
            {id: 'solve-grid-numbers', text: 'Click a clue number to jump to that clue. Stuck? Click any empty square for a quick peek at the letter.'}
        ]);
    });
}

var _crossingsCacheKey = _solveKey + '_crossings_v2';

function _getExpectedLength(enumStr) {
    var nums = (enumStr || '').match(/\d+/g);
    if (!nums) return 0;
    return nums.reduce(function(a, b) { return a + parseInt(b); }, 0);
}

function _checkCrossingConflict(input, guess) {
    var pattern = input.getAttribute('data-crossing') || '';
    if (!pattern) return null;
    var patLetters = pattern.replace(/-/g, '');
    var guessLetters = guess.replace(/\s/g, '').toUpperCase();
    for (var i = 0; i < patLetters.length && i < guessLetters.length; i++) {
        if (patLetters[i] !== '_' && patLetters[i] !== guessLetters[i]) {
            return {position: i + 1, expected: patLetters[i]};
        }
    }
    return null;
}

function _restoreCachedCrossings() {
    var cached = localStorage.getItem(_crossingsCacheKey);
    if (!cached) return false;
    try {
        var data = JSON.parse(cached);
        if (!data.patterns || Object.keys(data.patterns).length === 0) return false;
        // Only use cache if we have counts (not just empty patterns)
        if (!data.counts || Object.keys(data.counts).length === 0) return false;
        _applyCrossings(data.patterns, data.counts);
        return true;
    } catch(e) { return false; }
}

function _applyCrossings(patterns, counts) {
    document.querySelectorAll('.solve-input').forEach(function(row) {
        var input = row.querySelector('.solve-answer');
        var crossingEl = row.querySelector('.solve-crossing');
        var clueId = input.dataset.clueId;
        var pat = patterns[clueId];

        if (pat && !input.disabled) {
            input.placeholder = pat;
            input.setAttribute('data-crossing', pat);
            crossingEl.setAttribute('data-pattern', pat);
            crossingEl.classList.remove('hidden');
            var count = counts[clueId];
            if (count !== undefined) {
                crossingEl.textContent = count > 0 ? count + ' matches' : 'no matches';
            } else {
                crossingEl.textContent = '...';
                crossingEl.setAttribute('data-needs-count', pat.replace(/_/g, '?'));
            }
            window._hasCrossings = true;
        }
    });
    // Fetch any missing counts
    if (document.querySelectorAll('[data-needs-count]').length > 0) {
        setTimeout(_fetchMatchCountsStaggered, 300);
    }
}

function _fetchCrossings() {
    var solvedIds = _getSolvedIds();
    if (solvedIds.length === 0 && Object.keys(_revealedCells).length === 0) return;
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    var userAnswers = {};
    for (var k in state) {
        if (state[k].correct && state[k].value) {
            userAnswers[k] = state[k].value.replace(/\s/g, '').toUpperCase();
        }
    }
    var url = _puzzleUrl('crossings') + '?solved=' + solvedIds.join(',')
            + '&answers=' + encodeURIComponent(JSON.stringify(userAnswers));
    if (Object.keys(_revealedCells).length > 0) {
        url += '&revealed=' + encodeURIComponent(JSON.stringify(_revealedCells));
    }
    fetch(url).then(function(r) { return r.json(); }).then(function(crossings) {
        var patterns = {};
        document.querySelectorAll('.solve-input').forEach(function(row) {
            var input = row.querySelector('.solve-answer');
            var crossingEl = row.querySelector('.solve-crossing');
            var clueId = input.dataset.clueId;
            var card = input.closest('.clue-card');
            var linkedId = card && card.dataset.linkedId;

            var mainPat = crossings[clueId] || '';
            var linkedPat = (linkedId && crossings[linkedId]) || '';
            var pat = mainPat;
            if (linkedPat) {
                var raw = (mainPat || '') + (linkedPat || '');
                pat = _insertEnumBreaks(raw, input.dataset.enum || '');
            } else if (mainPat) {
                pat = _insertEnumBreaks(mainPat, input.dataset.enum || '');
            }

            if (pat && !input.disabled) {
                patterns[clueId] = pat;
                input.placeholder = pat;
                input.setAttribute('data-crossing', pat);
                var patternStr = pat.replace(/_/g, '?');
                crossingEl.textContent = '...';
                crossingEl.setAttribute('data-pattern', pat);
                crossingEl.classList.remove('hidden');
                crossingEl.setAttribute('data-needs-count', patternStr);
                window._hasCrossings = true;
            }
        });
        // Cache patterns for instant restore on refresh — clear stale counts
        localStorage.setItem(_crossingsCacheKey, JSON.stringify({patterns: patterns, counts: {}}));
        // Stagger match count fetches
        setTimeout(_fetchMatchCountsStaggered, 300);
    });
}

var _matchCountTimer = null;
function _fetchMatchCountsStaggered() {
    // Batch all pending counts into a single request
    if (_matchCountTimer) { clearTimeout(_matchCountTimer); _matchCountTimer = null; }
    var pending = document.querySelectorAll('[data-needs-count]');
    if (pending.length === 0) return;

    // Load cached patterns+counts to avoid re-requesting unchanged patterns
    var cached = {};
    try { cached = JSON.parse(localStorage.getItem(_crossingsCacheKey) || '{}'); } catch(e) { cached = {}; }
    if (!cached.counts) cached.counts = {};
    if (!cached.patterns) cached.patterns = {};

    var queries = {};
    pending.forEach(function(el) {
        var pat = el.getAttribute('data-needs-count');
        if (!pat) return;
        el.removeAttribute('data-needs-count');
        var row = el.closest('.solve-input');
        var input = row ? row.querySelector('.solve-answer') : null;
        var clueId = input ? input.dataset.clueId : null;
        if (!clueId || !pat) return;
        // Skip all-unknown patterns (no known letters)
        if (/^[?]+$/.test(pat.replace(/-/g, ''))) {
            el.textContent = '';
            el.classList.add('hidden');
            return;
        }
        queries[clueId] = {pattern: pat, enum: input.dataset.enum || ''};
    });

    if (Object.keys(queries).length === 0) return;

    fetch('/helper/pattern-counts?patterns=' + encodeURIComponent(JSON.stringify(queries)) + '&ht=' + _ht)
        .then(function(r) { return r.json(); })
        .then(function(counts) {
            var cached = {};
            try { cached = JSON.parse(localStorage.getItem(_crossingsCacheKey) || '{}'); } catch(e) { cached = {}; }
            if (!cached.counts) cached.counts = {};

            for (var clueId in counts) {
                var count = counts[clueId];
                cached.counts[clueId] = count;
                // Find the crossing element for this clue
                var input = document.querySelector('.solve-answer[data-clue-id="' + clueId + '"]');
                if (input) {
                    var crossEl = input.parentElement.querySelector('.solve-crossing');
                    if (crossEl) {
                        crossEl.textContent = count > 0 ? count + ' matches' : 'no matches';
                    }
                }
            }
            localStorage.setItem(_crossingsCacheKey, JSON.stringify(cached));
        });
}
function _cancelMatchCountFetches() {
    if (_matchCountTimer) { clearTimeout(_matchCountTimer); _matchCountTimer = null; }
    document.querySelectorAll('[data-needs-count]').forEach(function(el) {
        el.removeAttribute('data-needs-count');
    });
}

function _insertEnumBreaks(pat, enumStr) {
    // Insert dashes at word break positions: "BATTLEOFHASTINGS" + "(6,2,8)" -> "BATTLE-OF-HASTINGS"
    if (!enumStr) return pat;
    var nums = enumStr.match(/\d+/g);
    if (!nums) return pat;
    var total = nums.reduce(function(a, b) { return a + parseInt(b); }, 0);
    // Only insert breaks if total matches pattern length (without existing breaks)
    var rawPat = pat.replace(/-/g, '');
    if (rawPat.length !== total) return pat;
    if (nums.length <= 1) return rawPat;
    var result = '';
    var pos = 0;
    for (var i = 0; i < nums.length; i++) {
        var n = parseInt(nums[i]);
        result += rawPat.substring(pos, pos + n);
        pos += n;
        if (i < nums.length - 1) result += '-';
    }
    return result;
}

function _fetchMatchCount(el, patternStr, enumStr) {
    var url = '/helper/pattern-count?pattern=' + encodeURIComponent(patternStr) + '&ht=' + _ht;
    if (enumStr) url += '&enum=' + encodeURIComponent(enumStr);
    fetch(url)
        .then(function(r) { return r.text(); })
        .then(function(count) {
            count = parseInt(count.trim()) || 0;
            if (count > 0) {
                el.textContent = count + ' matches';
            } else {
                el.textContent = 'no matches';
            }
            // Cache the count
            var row = el.closest('.solve-input');
            if (row) {
                var input = row.querySelector('.solve-answer');
                if (input) {
                    try {
                        var cached = JSON.parse(localStorage.getItem(_crossingsCacheKey) || '{}');
                        if (!cached.counts) cached.counts = {};
                        cached.counts[input.dataset.clueId] = count;
                        localStorage.setItem(_crossingsCacheKey, JSON.stringify(cached));
                    } catch(e) {}
                }
            }
        });
}

var _lastFocusedClueId = null;
function solveInputFocus(input) {
    var clueId = input.dataset.clueId;
    _lastFocusedClueId = clueId;
}

function patternFromCrossing(el, crossingStr) {
    var pat = crossingStr || (el && el.getAttribute('data-pattern')) || '';
    if (!pat) return;
    _closeAllWordHelp();
    // Convert crossing format to pattern format: _ -> ?, keep letters and dashes
    var patternStr = pat.replace(/_/g, '?');
    // Get enum from the crossing element's sibling input
    var enumVal = '';
    if (el) {
        var solveRow = el.closest('.solve-input');
        if (solveRow) {
            var ansInput = solveRow.querySelector('.solve-answer');
            if (ansInput) enumVal = ansInput.dataset.enum || '';
        }
    }
    if (!enumVal) {
        var enumInput = document.getElementById('pattern-enum');
        if (enumInput) enumVal = enumInput.value || '';
    }
    // Set the enum input for subsequent form searches
    var enumInput = document.getElementById('pattern-enum');
    if (enumInput && enumVal) enumInput.value = enumVal;
    if (typeof _updatePatternEnumToggle === 'function') _updatePatternEnumToggle(enumVal);
    // Open tools overlay or floating panel with pattern tab
    var overlay = document.getElementById('tools-overlay');
    if (overlay && _solveMode) {
        var solveRow = el ? el.closest('.solve-input') : null;
        var ansInput = solveRow ? solveRow.querySelector('.solve-answer') : null;
        var clueId = ansInput ? ansInput.dataset.clueId : null;
        if (clueId && !_toolsClueId) openToolsOverlay(clueId);
    } else {
        var panel = document.getElementById('helper-panel');
        var openBtn = document.getElementById('helper-open');
        if (panel) panel.classList.remove('hidden');
        if (openBtn) openBtn.classList.add('hidden');
    }
    solverTab('pattern');
    var patInput = document.getElementById('pattern-input');
    patInput.value = patternStr;
    document.getElementById('pattern-include').value = '';
    // Search using _patternIncludeChanged which reads enum from the hidden input
    _patternIncludeChanged();
}

function saveAllToDb() {
    var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
    var answers = {};
    for (var k in state) {
        if (state[k].value) {
            answers[k] = state[k].value.replace(/\s/g, '').toUpperCase();
        }
    }
    var msg = document.getElementById('save-msg');
    msg.classList.remove('hidden');
    msg.className = 'text-xs text-orange-500';
    msg.textContent = 'Saving...';
    fetch('/admin/save-all-answers', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(answers)
    }).then(function(r) { return r.json(); }).then(function(data) {
        var text = data.saved + ' answer' + (data.saved === 1 ? '' : 's') + ' saved';
        if (data.skipped > 0) {
            text += ', ' + data.skipped + ' skipped \u2014 check lengths';
            msg.className = 'text-xs text-amber-600 font-medium';
        } else {
            msg.className = 'text-xs text-green-600 font-medium';
        }
        msg.textContent = text;
        setTimeout(function() { msg.classList.add('hidden'); }, 4000);
    }).catch(function() {
        msg.className = 'text-xs text-red-500';
        msg.textContent = 'Error saving';
    });
}

// Enter solve mode only if explicitly requested via ?solve=1, or restore if
// the user has genuine solve progress (answers they entered).
// Admin users: never auto-activate.
document.addEventListener('DOMContentLoaded', function() {
    if (_cfg.isAdmin) return;
    var explicitSolve = new URLSearchParams(window.location.search).get('solve') === '1';
    var wasActive = localStorage.getItem(_solveKey + '_active') === '1';
    var hasProgress = false;
    if (wasActive) {
        var state = JSON.parse(localStorage.getItem(_solveKey) || '{}');
        for (var k in state) {
            if (state[k].correct) { hasProgress = true; break; }
        }
    }
    if (_cfg.hasGrid && (explicitSolve || (wasActive && hasProgress))) {
        toggleSolveMode();
    } else {
        localStorage.removeItem(_solveKey + '_active');
    }

    // Block non-letter characters in solve inputs
    document.addEventListener('keypress', function(e) {
        if (!e.target.classList.contains('solve-answer')) return;
        if (e.target.disabled) return;
        var char = String.fromCharCode(e.which || e.keyCode);
        if (!/[a-zA-Z]/.test(char)) {
            e.preventDefault();
            var result = e.target.parentElement.querySelector('.solve-result');
            if (result && !result.textContent.trim()) {
                result.className = 'solve-result text-xs text-amber-600';
                result.textContent = 'Letters only';
                setTimeout(function() {
                    if (result.textContent === 'Letters only') {
                        result.textContent = '';
                        result.className = 'solve-result text-xs';
                    }
                }, 1500);
            }
        }
    });
});
