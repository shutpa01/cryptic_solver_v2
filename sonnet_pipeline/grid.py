"""Grid parser for Telegraph/Times crossword puzzles.

Parses the API JSON 'words' array into a grid model with intersection mapping.
Both Telegraph and Times use identical JSON structure:
  - copy.words[]: {id, x, y, solution?}
  - copy.clues[]: [{title: "Across"/"Down", clues: [{word, number, clue, format, length}]}]
  - copy.gridsize: {cols, rows}
"""

import re


class Grid:
    """15x15 crossword grid with word positions and intersection tracking."""

    def __init__(self, rows=15, cols=15):
        self.rows = rows
        self.cols = cols
        # word_id -> list of (row, col) cells
        self.word_cells = {}
        # word_id -> 'across' or 'down'
        self.word_direction = {}
        # word_id -> solution (if available)
        self.word_solution = {}
        # (row, col) -> {'across': word_id, 'down': word_id}
        self.cell_words = {}
        # word_id -> {clue_number, clue_text, format, length}
        self.word_clue = {}
        # Solved answers: word_id -> answer string
        self.solved = {}
        # Grid of known letters: (row, col) -> letter (uppercase)
        self.letters = {}

    def add_word(self, word_id, x_spec, y_spec, solution=None):
        """Parse word position from API format and add to grid.

        Across words: x="1-8", y="1"  (x is a range, y is fixed)
        Down words:   x="3",   y="1-9" (x is fixed, y is a range)
        """
        x_range = _parse_range(x_spec)
        y_range = _parse_range(y_spec)

        if len(x_range) > 1 and len(y_range) == 1:
            direction = 'across'
            cells = [(y_range[0] - 1, x - 1) for x in x_range]
        elif len(x_range) == 1 and len(y_range) > 1:
            direction = 'down'
            cells = [(y - 1, x_range[0] - 1) for y in y_range]
        else:
            # Single cell word (shouldn't happen in standard crosswords)
            direction = 'across'
            cells = [(y_range[0] - 1, x_range[0] - 1)]

        self.word_cells[word_id] = cells
        self.word_direction[word_id] = direction

        if solution:
            self.word_solution[word_id] = solution

        for cell in cells:
            if cell not in self.cell_words:
                self.cell_words[cell] = {}
            self.cell_words[cell][direction] = word_id

    def add_clue(self, word_id, number, clue_text, fmt, length):
        """Link a clue to its word."""
        self.word_clue[word_id] = {
            'number': number,
            'clue_text': clue_text,
            'format': fmt,
            'length': length,
        }

    def get_crossings(self, word_id):
        """Get intersection info for a word.

        Returns list of (position_in_word, crossing_word_id, position_in_crossing)
        """
        direction = self.word_direction[word_id]
        cross_dir = 'down' if direction == 'across' else 'across'
        crossings = []

        for i, cell in enumerate(self.word_cells[word_id]):
            if cross_dir in self.cell_words.get(cell, {}):
                cross_word_id = self.cell_words[cell][cross_dir]
                cross_cells = self.word_cells[cross_word_id]
                cross_pos = cross_cells.index(cell)
                crossings.append((i, cross_word_id, cross_pos))

        return crossings

    def set_answer(self, word_id, answer):
        """Fix an answer into the grid and update known letters."""
        clean = re.sub(r'[^A-Za-z]', '', answer).upper()
        cells = self.word_cells[word_id]
        if len(clean) != len(cells):
            return False

        self.solved[word_id] = clean
        for i, cell in enumerate(cells):
            self.letters[cell] = clean[i]
        return True

    def get_known_letters(self, word_id):
        """Get known letters for a word from crossing answers.

        Returns list of (position, letter) tuples.
        """
        known = []
        for i, cell in enumerate(self.word_cells[word_id]):
            if cell in self.letters:
                known.append((i, self.letters[cell]))
        return known

    def get_letter_pattern(self, word_id):
        """Get a pattern string like '?A??E??' for known/unknown letters."""
        cells = self.word_cells[word_id]
        return ''.join(
            self.letters.get(cell, '?')
            for cell in cells
        )

    def unsolved_word_ids(self):
        """Return word IDs not yet solved."""
        return [wid for wid in self.word_cells if wid not in self.solved]

    def summary(self):
        """Print grid state summary."""
        total = len(self.word_cells)
        solved = len(self.solved)
        lines = [f"Grid {self.cols}x{self.rows}: {total} words, {solved} solved"]
        for wid in sorted(self.word_cells.keys()):
            clue = self.word_clue.get(wid, {})
            num = clue.get('number', '?')
            direction = self.word_direction[wid]
            d_label = 'A' if direction == 'across' else 'D'
            fmt = clue.get('format', '?')
            pattern = self.get_letter_pattern(wid)

            if wid in self.solved:
                status = self.solved[wid]
            else:
                status = pattern

            lines.append(f"  {num:>2}{d_label} ({fmt:>5}) {status}")
        return '\n'.join(lines)


def _parse_range(spec):
    """Parse '1-8' -> [1,2,3,4,5,6,7,8] or '3' -> [3]."""
    spec = str(spec).strip()
    if '-' in spec:
        parts = spec.split('-')
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    return [int(spec)]


def load_grid_from_json(puzzle_data):
    """Build a Grid from Telegraph/Times API JSON.

    Args:
        puzzle_data: The parsed JSON dict. Can be either:
          - The full API response (with 'json' or 'data' wrapper)
          - The 'copy' dict directly
    """
    # Navigate to the 'copy' section
    if 'json' in puzzle_data:
        copy = puzzle_data['json']['copy']  # Telegraph format
    elif 'data' in puzzle_data:
        copy = puzzle_data['data']['copy']  # Times format
    elif 'copy' in puzzle_data:
        copy = puzzle_data['copy']
    elif 'words' in puzzle_data:
        copy = puzzle_data  # Already at the right level
    else:
        raise ValueError(f"Cannot find puzzle data. Top-level keys: {list(puzzle_data.keys())}")

    gridsize = copy.get('gridsize', {})
    rows = int(gridsize.get('rows', 15))
    cols = int(gridsize.get('cols', 15))

    grid = Grid(rows=rows, cols=cols)

    # Add words
    for word in copy.get('words', []):
        grid.add_word(
            word_id=word['id'],
            x_spec=word['x'],
            y_spec=word['y'],
            solution=word.get('solution'),
        )

    # Add clues
    for group in copy.get('clues', []):
        for clue in group.get('clues', []):
            # Clean HTML entities from clue text
            import html
            clue_text = html.unescape(clue.get('clue', ''))
            clue_text = re.sub(r'<[^>]+>', '', clue_text)

            grid.add_clue(
                word_id=clue['word'],
                number=clue['number'],
                clue_text=clue_text,
                fmt=clue.get('format', ''),
                length=clue.get('length', 0),
            )

    return grid
