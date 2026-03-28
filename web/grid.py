"""Crossword grid construction and reconstruction.

Two paths to produce a grid:
1. parse_grid_solution() — direct parse of a flat solution string
   (e.g. from the Telegraph API).  Fast and exact.
2. reconstruct_grid() — algorithmic reconstruction from clue answers
   when no solution string is available.

Standard British cryptic grids are 15x15 with 180-degree rotational symmetry.
The reconstruction algorithm works by:
1. Grouping across clues into rows
2. Trying symmetric row assignments (standard every-other-row first,
   then other symmetric spacings)
3. Searching for word placements with symmetry pairing
4. Placing across words, then finding each down word's column via
   letter matching constrained by reading-order numbering
"""

import json
import re
import sys
from itertools import combinations, product
from pathlib import Path

# Allow importing from scraper module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Path 1: Direct parse from solution string
# ---------------------------------------------------------------------------

def parse_grid_solution(solution, rows=15, cols=15):
    """Build a grid from a flat solution string.

    Args:
        solution: string of length rows*cols.  Letters are cell content,
                  spaces represent black squares.
        rows: grid height (default 15)
        cols: grid width (default 15)

    Returns:
        dict with 'cells' (rows x cols), 'rows', 'cols' — same format as
        reconstruct_grid().  Returns None if the solution string is invalid.
    """
    if not solution or len(solution) != rows * cols:
        return None

    # Build 2D letter grid (None = black)
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            ch = solution[r * cols + c]
            row.append(ch.upper() if ch != ' ' else None)
        grid.append(row)

    # Assign clue numbers using standard crossword rules:
    # A cell gets a number if it starts an across entry OR a down entry.
    # Across start: letter cell, left is edge/black, at least one letter to right.
    # Down start: letter cell, above is edge/black, at least one letter below.
    number = 1
    num_at = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                continue
            starts_across = (
                (c == 0 or grid[r][c - 1] is None)
                and c + 1 < cols and grid[r][c + 1] is not None
            )
            starts_down = (
                (r == 0 or grid[r - 1][c] is None)
                and r + 1 < rows and grid[r + 1][c] is not None
            )
            if starts_across or starts_down:
                num_at[(r, c)] = number
                number += 1

    # Build output cells
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if grid[r][c] is None:
                row.append(None)
            else:
                cell = {"letter": grid[r][c]}
                if (r, c) in num_at:
                    cell["number"] = num_at[(r, c)]
                row.append(cell)
        cells.append(row)

    return {"cells": cells, "rows": rows, "cols": cols}


# ---------------------------------------------------------------------------
# Path 1b: Live rebuild from JSON structure + current DB answers
# ---------------------------------------------------------------------------

def build_grid_from_json(source, puzzle_number, clue_data):
    """Build a grid live from JSON structure + current DB answers.

    Returns grid dict or None if no JSON available.
    """
    from scraper.danword.danword_lookup import find_puzzle_json
    json_path = find_puzzle_json(source, puzzle_number)
    if json_path is None:
        return None

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "json" in data:
        copy = data["json"].get("copy", {})
        grid_array = data["json"].get("grid")
    elif "data" in data:
        copy = data["data"].get("copy", {})
        grid_array = None
    else:
        copy = data.get("copy", {})
        grid_array = None

    gridsize = copy.get("gridsize", {})
    cols = int(gridsize.get("cols", 15))
    rows = int(gridsize.get("rows", 15))
    words = copy.get("words", [])
    clues_sections = copy.get("clues", [])

    if not words or not clues_sections:
        return None

    # Reuse helpers from danword_lookup for parsing word positions
    # Note: _build_word_cells returns 1-indexed coords; convert to 0-indexed
    from scraper.danword.danword_lookup import _build_word_cells, _build_clue_to_word
    word_cells_1 = _build_word_cells(words)
    word_cells = {
        wid: [(r - 1, c - 1) for r, c in cells]
        for wid, cells in word_cells_1.items()
    }
    clue_to_word_raw = _build_clue_to_word(clues_sections)
    clue_to_word = {}
    for (num_str, direction), wid in clue_to_word_raw.items():
        clue_to_word[(int(num_str), direction)] = wid

    # Determine black vs white cells (0-indexed)
    black_cells = set()
    white_cells = set()
    if grid_array:
        for r_idx, row in enumerate(grid_array):
            for c_idx, cell in enumerate(row):
                if cell.get("Blank") == "blank":
                    black_cells.add((r_idx, c_idx))
                else:
                    white_cells.add((r_idx, c_idx))
    else:
        for cells in word_cells.values():
            for cell in cells:
                white_cells.add(cell)

    # Place current DB answers into grid
    grid_letters = {}
    clue_answers = {}
    if clue_data:
        for c in clue_data:
            num = int(c["clue_number"])
            direction = c["direction"]
            answer = c.get("answer") or ""
            if answer:
                clue_answers[(num, direction)] = answer

    # Build spanning clue map from JSON "links" field
    _spanning_links = {}  # (main_num, main_dir) -> [(linked_num, linked_dir), ...]
    for section in clues_sections:
        sec_dir = "across" if section.get("title", "").lower().startswith("across") else "down"
        for clue_entry in section.get("clues", []):
            links = clue_entry.get("links", [])
            if links:
                main_key = (int(clue_entry["number"]), sec_dir)
                _spanning_links[main_key] = [
                    (int(lk["number"]), lk["direction"].lower())
                    for lk in links
                ]

    for (num, direction), answer in clue_answers.items():
        wid = clue_to_word.get((num, direction))
        if wid is None:
            continue
        cells = word_cells.get(wid, [])
        clean_ans = re.sub(r"[^A-Za-z]", "", answer).upper()
        if len(clean_ans) == len(cells):
            for i, (row, col) in enumerate(cells):
                grid_letters[(row, col)] = clean_ans[i]
        elif (num, direction) in _spanning_links:
            # Spanning clue: place letters in main cells, then linked cells in order
            all_cells = list(cells)
            for linked_num, linked_dir in _spanning_links[(num, direction)]:
                linked_wid = clue_to_word.get((linked_num, linked_dir))
                if linked_wid:
                    all_cells.extend(word_cells.get(linked_wid, []))
            if len(clean_ans) == len(all_cells):
                for i, (row, col) in enumerate(all_cells):
                    grid_letters[(row, col)] = clean_ans[i]

    # Assign clue numbers using standard rules
    letter_grid = [[None] * cols for _ in range(rows)]
    for (r, c), letter in grid_letters.items():
        if 0 <= r < rows and 0 <= c < cols:
            letter_grid[r][c] = letter

    # Mark all white cells (even without answers) so numbering is correct
    for r, c in white_cells:
        if 0 <= r < rows and 0 <= c < cols and letter_grid[r][c] is None:
            letter_grid[r][c] = ""  # white but no letter yet

    number = 1
    num_at = {}
    for r in range(rows):
        for c in range(cols):
            if letter_grid[r][c] is None:
                continue
            is_white = (r, c) in white_cells or (r, c) in grid_letters
            if not is_white:
                continue
            left_black = c == 0 or letter_grid[r][c - 1] is None
            above_black = r == 0 or letter_grid[r - 1][c] is None
            right_white = c + 1 < cols and letter_grid[r][c + 1] is not None
            below_white = r + 1 < rows and letter_grid[r + 1][c] is not None
            starts_across = left_black and right_white
            starts_down = above_black and below_white
            if starts_across or starts_down:
                num_at[(r, c)] = number
                number += 1

    # Build output cells
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if letter_grid[r][c] is None:
                row.append(None)
            else:
                letter = letter_grid[r][c]
                cell = {"letter": letter if letter else ""}
                if (r, c) in num_at:
                    cell["number"] = num_at[(r, c)]
                row.append(cell)
        cells.append(row)

    return {"cells": cells, "rows": rows, "cols": cols}


# ---------------------------------------------------------------------------
# Path 2: Algorithmic reconstruction from clue answers
# ---------------------------------------------------------------------------

def reconstruct_grid(clue_data, size=15):
    """Reconstruct a crossword grid from clue data.

    Args:
        clue_data: list of dicts with keys: clue_number, direction, answer
        size: grid size (default 15)

    Returns:
        dict with 'cells' (size x size list, each cell = {'letter','number'}
        or None), 'rows', 'cols' — or None if reconstruction fails.
    """
    across = {}
    down = {}
    for c in clue_data:
        num = int(c["clue_number"])
        answer = re.sub(r"[^A-Za-z]", "", c["answer"]).upper()
        if c["direction"] == "across":
            across[num] = answer
        else:
            down[num] = answer

    if not across or not down:
        return None

    all_nums = sorted(set(across.keys()) | set(down.keys()))

    # Try the greedy grouping first, then splits of multi-word groups
    for row_groups in _candidate_groupings(across, all_nums, size):
        n_rows = len(row_groups)
        if n_rows > size:
            continue

        for grid_rows in _symmetric_row_assignments(n_rows, size):
            # Quick feasibility: can long down words fit?
            if not _check_down_feasibility(down, all_nums, across,
                                           row_groups, grid_rows, size):
                continue

            result = _try_symmetric_placements(
                size, across, down, all_nums,
                row_groups, grid_rows, n_rows)
            if result is not None:
                return result

    return None


def _candidate_groupings(across, all_nums, size):
    """Yield across groupings: greedy first, then with splits and re-merges."""
    across_nums = sorted(across.keys())
    greedy = _group_across_into_rows(across_nums, across, all_nums, size)
    seen = set()

    def _key(groups):
        return tuple(tuple(g) for g in groups)

    def _emit(groups):
        k = _key(groups)
        if k not in seen:
            seen.add(k)
            return True
        return False

    if _emit(greedy):
        yield greedy

    # Identify which groups can be split (have 2+ words)
    splittable = [i for i, g in enumerate(greedy) if len(g) > 1]

    # Try splitting 1 group at a time
    for idx in splittable:
        alt = list(greedy)
        group = alt[idx]
        replacement = [[(num, length)] for num, length in group]
        alt = alt[:idx] + replacement + alt[idx + 1:]
        if _emit(alt):
            yield alt

    # Try splitting 2 groups, then also re-merge adjacent single-word groups
    for i, j in combinations(splittable, 2):
        alt = list(greedy)
        group_j = alt[j]
        rep_j = [[(num, length)] for num, length in group_j]
        alt = alt[:j] + rep_j + alt[j + 1:]
        group_i = alt[i]
        rep_i = [[(num, length)] for num, length in group_i]
        alt = alt[:i] + rep_i + alt[i + 1:]
        if _emit(alt):
            yield alt

        # Re-merge: try merging each pair of adjacent single-word groups
        for k in range(len(alt) - 1):
            g1, g2 = alt[k], alt[k + 1]
            if len(g1) == 1 and len(g2) == 1:
                merged_width = g1[0][1] + 1 + g2[0][1]
                if merged_width <= size:
                    last_num = g1[-1][0]
                    first_num = g2[0][0]
                    between = [n for n in all_nums
                               if last_num < n < first_num]
                    if all(n not in across for n in between):
                        remerged = alt[:k] + [g1 + g2] + alt[k + 2:]
                        if _emit(remerged):
                            yield remerged


def _check_down_feasibility(down, all_nums, across, row_groups,
                             grid_rows, size):
    """Quick check: can all down words physically fit?

    A down word numbered N must start at or after the row of the most
    recent across word before N.  Verify start_row + length <= size.
    """
    across_row = {}
    for grid_row, group in zip(grid_rows, row_groups):
        for num, _ in group:
            across_row[num] = grid_row

    for num in sorted(down.keys()):
        wlen = len(down[num])
        min_start = 0
        for n in all_nums:
            if n < num and n in across_row:
                min_start = across_row[n]
        if min_start + wlen > size:
            return False
    return True


def _symmetric_row_assignments(n_rows, size=15):
    """Generate symmetric row assignments sorted by spacing uniformity.

    For n_rows groups, generates all ways to place them on a grid of `size`
    rows respecting 180-degree rotational symmetry.  The standard
    every-other-row pattern (e.g., [0,2,4,6,8,10,12,14] for 8 groups)
    comes first.
    """
    half = n_rows // 2
    is_odd = n_rows % 2 == 1
    mid = size // 2

    results = []
    for upper in combinations(range(mid), half):
        rows = list(upper)
        if is_odd:
            rows.append(mid)
        rows.extend(size - 1 - r for r in reversed(list(upper)))
        results.append(tuple(rows))

    def spacing_variance(rows):
        if len(rows) < 2:
            return 0
        gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        avg = sum(gaps) / len(gaps)
        return sum((g - avg) ** 2 for g in gaps)

    results.sort(key=spacing_variance)
    return results


# ---------------------------------------------------------------------------
# Placement-based search (replaces offset-only approach)
# ---------------------------------------------------------------------------

def _row_placements(word_lengths, size):
    """Generate all valid starting-column tuples for words on a row.

    Distributes available space among: leading gap, inter-word gaps (≥1 each),
    and trailing gap.  Returns list of tuples, one start column per word.
    """
    n = len(word_lengths)
    results = []

    def _place(idx, min_col, starts):
        if idx == n:
            results.append(tuple(starts))
            return
        wlen = word_lengths[idx]
        # Leave room for remaining words (each needs length + 1 gap)
        remaining = sum(word_lengths[idx + 1:]) + (n - idx - 1)
        max_start = size - wlen - remaining
        for col in range(min_col, max_start + 1):
            _place(idx + 1, col + wlen + 1, starts + [col])

    _place(0, 0, [])
    return results


def _make_row_pattern(starts, word_lengths, size):
    """Create a boolean pattern (True=letter, False=black) from placements."""
    pattern = [False] * size
    for s, wl in zip(starts, word_lengths):
        for k in range(wl):
            pattern[s + k] = True
    return tuple(pattern)


def _try_symmetric_placements(size, across, down, all_nums,
                               row_groups, grid_rows, n_rows):
    """Try symmetric placement combinations for a given row assignment.

    For each pair of symmetric rows, finds placements whose black-cell
    patterns are 180-degree mirrors.  Then tries all compatible combinations.
    """
    # Compute all valid placements per row group
    all_placements = []
    all_word_lengths = []
    for group in row_groups:
        wlens = [length for _, length in group]
        all_word_lengths.append(wlens)
        all_placements.append(_row_placements(wlens, size))

    # Identify symmetric pairs and middle row
    paired = {}
    mid_row = None
    pair_keys = []
    for i in range(n_rows):
        j = n_rows - 1 - i
        if i < j:
            paired[i] = j
            pair_keys.append(i)
        elif i == j:
            mid_row = i

    # For each pair, find compatible placement pairs (reversed pattern match)
    pair_options = {}
    for i in pair_keys:
        j = paired[i]
        options = []
        # Pre-compute patterns for row j
        j_patterns = {}
        for pj in all_placements[j]:
            j_patterns[pj] = _make_row_pattern(pj, all_word_lengths[j], size)

        for pi in all_placements[i]:
            pat_i = _make_row_pattern(pi, all_word_lengths[i], size)
            pat_i_rev = pat_i[::-1]
            for pj, pat_j in j_patterns.items():
                if pat_i_rev == pat_j:
                    options.append((pi, pj))
        if not options:
            return None
        pair_options[i] = options

    # For middle row, find self-symmetric placements
    mid_options = None
    if mid_row is not None:
        mid_options = []
        for p in all_placements[mid_row]:
            pat = _make_row_pattern(p, all_word_lengths[mid_row], size)
            if pat[::-1] == pat:
                mid_options.append(p)
        if not mid_options:
            return None

    # Cap total combinations
    total = 1
    for k in pair_keys:
        total *= len(pair_options[k])
    if mid_options:
        total *= len(mid_options)
    if total > 50_000:
        return None

    # Enumerate all compatible combinations
    option_lists = [pair_options[k] for k in pair_keys]
    for combo in product(*option_lists):
        placements = [None] * n_rows
        for idx, k in enumerate(pair_keys):
            pi, pj = combo[idx]
            placements[k] = pi
            placements[paired[k]] = pj

        if mid_options is not None:
            for mid_p in mid_options:
                placements[mid_row] = mid_p
                result = _try_layout(size, across, down, all_nums,
                                     row_groups, grid_rows, placements)
                if result is not None:
                    return result
        else:
            result = _try_layout(size, across, down, all_nums,
                                 row_groups, grid_rows, placements)
            if result is not None:
                return result

    return None


def _try_layout(size, across, down, all_nums, row_groups, grid_rows,
                placements):
    """Try a specific placement layout.  Return grid dict or None.

    placements: list of tuples, one per row group.  Each tuple contains
    the starting column for each word in that group.
    """
    grid = [[None] * size for _ in range(size)]

    # Place across words using the specified starting columns
    across_starts = {}
    for i, (grid_row, group) in enumerate(zip(grid_rows, row_groups)):
        starts = placements[i]
        for j, (num, length) in enumerate(group):
            col = starts[j]
            if col + length > size:
                return None
            word = across[num]
            for k, letter in enumerate(word):
                grid[grid_row][col + k] = letter
            across_starts[num] = (grid_row, col)

    known_positions = dict(across_starts)
    down_positions = {}

    for num in sorted(down.keys()):
        word = down[num]

        # Reading-order bounds
        after_pos = (-1, -1)
        before_pos = (size, size)
        for n in all_nums:
            if n < num and n in known_positions:
                pos = known_positions[n]
                if pos > after_pos:
                    after_pos = pos
            elif n > num and n in known_positions:
                pos = known_positions[n]
                if pos < before_pos:
                    before_pos = pos

        best = _find_down_column(word, grid, size, after_pos, before_pos)
        if best is None:
            return None
        down_positions[num] = best
        known_positions[num] = best

        # Place down word letters
        for i, letter in enumerate(word):
            r = best[0] + i
            existing = grid[r][best[1]]
            if existing is not None and existing != letter:
                return None
            grid[r][best[1]] = letter

    # Build number positions and verify reading order
    number_pos = {}
    for num, pos in across_starts.items():
        number_pos[num] = pos
    for num, pos in down_positions.items():
        if num not in number_pos:
            number_pos[num] = pos

    positions = sorted(number_pos.items(), key=lambda x: (x[1][0], x[1][1]))
    expected = sorted(all_nums)
    if [p[0] for p in positions] != expected:
        return None

    # Build output cells
    num_at = {(r, c): num for num, (r, c) in number_pos.items()}
    cells = []
    for r in range(size):
        row = []
        for c in range(size):
            if grid[r][c] is None:
                row.append(None)
            else:
                cell = {"letter": grid[r][c]}
                if (r, c) in num_at:
                    cell["number"] = num_at[(r, c)]
                row.append(cell)
        cells.append(row)

    return {"cells": cells, "rows": size, "cols": size}


def _find_down_column(word, grid, size, after_pos, before_pos):
    """Find the starting position (row, col) for a down word.

    Uses letter intersection matching constrained by reading-order bounds.
    """
    wlen = len(word)
    candidates = []

    for start_row in range(size):
        if start_row + wlen > size:
            break

        for col in range(size):
            pos = (start_row, col)
            if pos <= after_pos or pos >= before_pos:
                continue

            # Cell above must be black or top edge
            if start_row > 0 and grid[start_row - 1][col] is not None:
                continue

            match = True
            crossing_count = 0
            for i, letter in enumerate(word):
                r = start_row + i
                existing = grid[r][col]
                if existing is not None:
                    if existing != letter:
                        match = False
                        break
                    crossing_count += 1

            if match and crossing_count >= 2:
                candidates.append((start_row, col, crossing_count))

    if not candidates:
        return None

    # Prefer most crossings, then earliest in reading order
    candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
    return (candidates[0][0], candidates[0][1])


def _group_across_into_rows(across_nums, across, all_nums, size):
    """Group across clues into rows based on width constraints (greedy)."""
    row_groups = []
    current = []
    current_width = 0

    for anum in across_nums:
        alen = len(across[anum])

        if not current:
            current = [(anum, alen)]
            current_width = alen
            continue

        last_num = current[-1][0]
        between = [n for n in all_nums if last_num < n < anum]
        all_down_only = all(n not in across for n in between)

        new_width = current_width + 1 + alen

        if all_down_only and new_width <= size:
            current.append((anum, alen))
            current_width = new_width
        else:
            row_groups.append(current)
            current = [(anum, alen)]
            current_width = alen

    if current:
        row_groups.append(current)

    return row_groups
