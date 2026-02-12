#!/usr/bin/env python3
"""
Compound wordplay analysis for cryptic crossword solver.

This file analyzes cases where anagram detection finds partial matches
that need additional wordplay techniques (substitution, reversal, etc.)
to complete the construction.

Architecture:
1. Run original pipeline simulator (untouched)
2. Filter for clues with anagram hits but remaining unused words
3. Apply ExplanationSystemBuilder for proper word attribution
4. Show compound wordplay opportunities with database-backed explanations

UPDATED: Now uses database-integrated CompoundWordplayAnalyzer
UPDATED: Now persists evidence and compound stages to SQLite
"""

import sys

# Add project root to path

# Import the original pipeline simulator (maintaining sanctity)
from pipeline_simulator import run_pipeline_probe, MAX_CLUES, WORDPLAY_TYPE
from stages.explanation import ExplanationSystemBuilder

# Import EvidenceAnalyzer to apply evidence ranking to compound candidates
from evidence_analysis import EvidenceAnalyzer

# Pipeline Persistence Integration
try:
    from persistence import save_stage, get_stage_summary

    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    print("WARNING: Pipeline persistence not available for compound analysis")


class CompoundAnalyzer:
    """Analyzes compound wordplay opportunities using proper word attribution."""

    def __init__(self):
        """Initialize the explanation system builder."""
        self.explanation_builder = None
        try:
            self.explanation_builder = ExplanationSystemBuilder()
            print("Explanation system builder loaded successfully.")
        except Exception as e:
            print(f"WARNING: Explanation builder failed to load: {e}")
            self.explanation_builder = None

    def close(self):
        """Clean up resources."""
        if self.explanation_builder:
            self.explanation_builder.close()

    def is_compound_candidate(self, record):
        """
        Check if a clue is a compound candidate:
        - Has anagram hits (partial or complete)
        - Still has meaningful unused words for additional wordplay
        """
        summary = record.get("summary", {})
        has_anagram_hits = summary.get("anagram_hits", 0) > 0

        if not has_anagram_hits:
            return False

        # Check if anagram hits have meaningful unused words
        anagrams = record.get('anagrams', [])
        for anagram_hit in anagrams:
            unused_words = anagram_hit.get('unused_words', [])
            meaningful_unused = [w for w in unused_words
                                 if len(w) > 2 and not w.replace(',', '').replace('-',
                                                                                  '').isdigit()]
            if len(meaningful_unused) >= 1:  # At least 1 meaningful remaining word
                return True

        return False

    def analyze_compound_cohort(self, results, run_id=None):
        """
        Analyze compound candidates using evidence-ranked results.
        UPDATED: Now works with evidence-analyzed results and database lookups.
        UPDATED: Now persists evidence and compound stages.
        """
        compound_candidates = [r for r in results if self.is_compound_candidate(r)]

        print(f"\n🧩 COMPOUND WORDPLAY COHORT ANALYSIS:")
        print(f"Total clues processed: {len(results)}")
        print(
            f"Compound candidates (anagram hits with remaining words): {len(compound_candidates)}")

        if not compound_candidates:
            print("No compound candidates found.")
            return []

        # Apply evidence analysis to get ranked candidates for each clue
        evidence_analyzer = EvidenceAnalyzer()
        evidence_enhanced_results = []

        print("Applying evidence analysis to compound candidates...")
        for i, record in enumerate(compound_candidates):
            try:
                # Apply evidence analysis to get complete ranked candidate information
                enhanced_record = evidence_analyzer.apply_evidence_scoring(record,
                                                                           debug=False)
                evidence_enhanced_results.append(enhanced_record)

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1} compound candidates...")

            except Exception as e:
                print(f"Warning: Could not analyze compound candidate {i + 1}: {e}")
                continue

        # ---- SAVE EVIDENCE STAGE ----
        if run_id is not None and PERSISTENCE_AVAILABLE:
            save_stage('evidence', run_id, evidence_enhanced_results)

        # Now work with evidence-enhanced results for compound analysis
        if not self.explanation_builder:
            print("\nCompound analysis disabled - explanation builder not available.")
            return []

        # Use ExplanationSystemBuilder with evidence-enhanced data
        try:
            # Step 1: Apply enhance_pipeline_data to add definition windows and word attribution
            # This now queries the database for indicators and substitutions
            enhanced_cases = self.explanation_builder.enhance_pipeline_data(
                evidence_enhanced_results)

            # Step 2: Build systematic explanations from enhanced cases
            explanations = self.explanation_builder.build_explanations(enhanced_cases)

            # Step 3: Include ALL analyzed cases (not just compound)
            # Quality rating will distinguish pure anagrams from compounds
            compound_explanations = []
            for i, exp in enumerate(explanations):
                # Get enhanced case data
                remaining = enhanced_cases[i].get('remaining_unresolved', [])
                compound_sol = enhanced_cases[i].get('compound_solution', {})

                # Merge enhanced case data with explanation
                exp['compound_solution'] = compound_sol
                exp['remaining_unresolved'] = remaining
                exp['word_roles'] = enhanced_cases[i].get('word_roles', [])
                exp['id'] = evidence_enhanced_results[i].get('id')  # Preserve clue_id

                # CRITICAL: Copy likely_answer and db_answer for display
                exp['likely_answer'] = enhanced_cases[i].get('likely_answer', '')
                exp['db_answer'] = enhanced_cases[i].get('db_answer', '')
                exp['answer_matches'] = enhanced_cases[i].get('answer_matches', False)
                exp['anagram_component'] = enhanced_cases[i].get('anagram_component', {})
                exp['definition_window'] = enhanced_cases[i].get('definition_window')

                compound_explanations.append(exp)

            print(f"Cases with evidence analysis: {len(evidence_enhanced_results)}")
            print(f"Cases with enhanced attribution: {len(enhanced_cases)}")
            print(f"Cases analyzed: {len(compound_explanations)}")

            # ---- SAVE COMPOUND STAGE ----
            if run_id is not None and PERSISTENCE_AVAILABLE:
                save_stage('compound', run_id, compound_explanations)

            return compound_explanations

        except Exception as e:
            print(f"Error in compound analysis: {e}")
            import traceback
            traceback.print_exc()
            return []


def display_compound_results(compound_results, max_display=10):
    """Display compound wordplay analysis results with new format."""

    if not compound_results:
        print("\nNo compound wordplay results to display.")
        return

    # Sort by quality - LOW FIRST for debugging
    quality_order = {'solved': 4, 'high': 3, 'medium': 2, 'low': 1, 'none': 0}
    sorted_results = sorted(compound_results,
                            key=lambda x: quality_order.get(
                                x.get('explanation', {}).get('quality', 'none'), 0
                            ), reverse=False)  # Low first

    print(f"\n🧩 COMPOUND WORDPLAY ANALYSIS RESULTS (Top {max_display}):")
    print("=" * 80)

    for i, result in enumerate(sorted_results[:max_display], 1):
        print(f"\n[{i}] CLUE: {result['clue']}")

        # Show LIKELY ANSWER (what we solved) and DB ANSWER (for comparison)
        likely_answer = result.get('likely_answer', result.get('answer', ''))
        db_answer = result.get('db_answer', result.get('answer', ''))
        answer_matches = result.get('answer_matches', likely_answer == db_answer)

        match_symbol = "✓" if answer_matches else "✗"
        print(f"    LIKELY ANSWER: {likely_answer}")
        print(f"    DB ANSWER:     {db_answer} {match_symbol}")

        # Get explanation
        explanation = result.get('explanation', {})
        quality = explanation.get('quality', 'unknown')
        formula = explanation.get('formula', 'No formula')
        breakdown = explanation.get('breakdown', [])

        print(f"    QUALITY: {quality}")
        print(f"\n    WORDPLAY: {formula}")

        if breakdown:
            print(f"\n    BREAKDOWN:")
            for line in breakdown:
                print(f"      {line}")

        # Compound solution details
        compound_sol = result.get('compound_solution', {})
        if compound_sol:
            print(f"\n    COMPOUND SOLUTION:")

            if compound_sol.get('substitutions'):
                print(f"      Substitutions found:")
                for word, letters, category in compound_sol['substitutions']:
                    print(f"        • {word} → {letters} ({category})")

            if compound_sol.get('operation_indicators'):
                print(f"      Operation indicators:")
                for word, op_type, subtype in compound_sol['operation_indicators']:
                    sub_str = f"/{subtype}" if subtype else ""
                    print(f"        • {word} → {op_type}{sub_str}")

            if compound_sol.get('positional_indicators'):
                print(
                    f"      Positional indicators: {compound_sol['positional_indicators']}")

            if compound_sol.get('construction'):
                constr = compound_sol['construction']
                print(f"      Construction: {constr.get('operation', 'unknown')}")

            # Only show as resolved if answer is correct
            fully_resolved = compound_sol.get('fully_resolved', False)
            if result.get('explanation', {}).get('quality') == 'INCORRECT':
                fully_resolved = False
            print(f"      Fully resolved: {fully_resolved}")

        # Show unaccounted indicator/fodder for self-learning
        if compound_sol:
            letters_needed = compound_sol.get('letters_still_needed', '')
            unresolved_words = compound_sol.get('unresolved_words', [])
            if letters_needed and unresolved_words:
                print(f"    Indicator and fodder unaccounted for: {letters_needed.upper()} from {unresolved_words}")
            elif letters_needed:
                print(f"    Letters unaccounted for: {letters_needed.upper()}")

        # Remaining unresolved (legacy field)
        remaining = result.get('remaining_unresolved', [])
        if remaining:
            print(f"    STILL UNRESOLVED: {remaining}")

        print("-" * 80)

    # Summary statistics
    print(f"\n📊 SUMMARY:")
    quality_counts = {}
    for r in compound_results:
        q = r.get('explanation', {}).get('quality', 'none')
        quality_counts[q] = quality_counts.get(q, 0) + 1

    for q in ['solved', 'high', 'medium', 'low', 'none']:
        if q in quality_counts:
            print(f"  {q}: {quality_counts[q]}")

    # Count fully resolved (only correct answers count)
    fully_resolved = sum(1 for r in compound_results
                         if
                         (r.get('compound_solution') or {}).get('fully_resolved', False)
                         and r.get('explanation', {}).get('quality') != 'INCORRECT')
    print(f"\n  Fully resolved: {fully_resolved}/{len(compound_results)}")

    # Count answer matches
    matches = sum(1 for r in compound_results
                  if r.get('answer_matches', False) or
                  r.get('likely_answer', '') == r.get('db_answer', ''))
    print(f"  Answer matches: {matches}/{len(compound_results)}")


def main():
    """Main analysis function."""
    print("🧩 COMPOUND WORDPLAY ANALYSIS")
    print("=" * 60)
    print("Maintaining absolute sanctity of original pipeline simulator")
    print("Using database-backed indicators and substitutions")
    print("=" * 60)

    # Initialize compound analyzer
    analyzer = CompoundAnalyzer()

    # Step 1: Run original pipeline simulator
    print("\n📋 STEP 1: Running original pipeline simulator...")

    # Override the ONLY_MISSING_DEFINITION setting for analysis
    # We need clues where the answer IS in definition candidates
    import pipeline_simulator
    original_setting = pipeline_simulator.ONLY_MISSING_DEFINITION
    pipeline_simulator.ONLY_MISSING_DEFINITION = False  # We want answer in def candidates

    try:
        results, overall = run_pipeline_probe()

        # Restore original setting
        pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting

    except Exception as e:
        # Restore original setting even if error occurs
        pipeline_simulator.ONLY_MISSING_DEFINITION = original_setting
        analyzer.close()
        raise e

    # Get run_id from persistence (pipeline_simulator just saved input/definition/anagram stages)
    run_id = None
    if PERSISTENCE_AVAILABLE:
        try:
            summary = get_stage_summary()
            run_id = summary.get('run_id')
            print(f"\n📊 Pipeline persistence: run_id={run_id}")
        except Exception as e:
            print(f"WARNING: Could not get run_id: {e}")

    # Show original results summary
    print("\n📊 ORIGINAL PIPELINE RESULTS:")
    print(f"  clues processed           : {overall['clues']}")
    print(f"  gate failed (no def match): {overall.get('gate_failed', 0)}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']}")

    # Step 2: Analyze compound wordplay cohort
    print("\n🧩 STEP 2: Analyzing compound wordplay cohort with database lookups...")
    enhanced_results = analyzer.analyze_compound_cohort(results, run_id=run_id)

    # Step 3: Display compound analysis results
    if enhanced_results:
        display_compound_results(enhanced_results, max_display=20)

    # Clean up
    analyzer.close()

    print("\n✅ Analysis complete. Original pipeline simulator untouched.")


if __name__ == "__main__":
    main()