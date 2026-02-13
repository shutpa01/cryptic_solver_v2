#!/usr/bin/env python3
"""
Evidence-based scoring analysis for successful anagram clues.

This file maintains the absolute sanctity of the original pipeline simulator
by calling it first, then only applying evidence scoring via the permanent engine.

Architecture:
1. Run original pipeline simulator (untouched)
2. Filter for successful anagram clues
3. Apply evidence scoring through permanent engine only
4. Show ranking improvements
"""


# Add project root to path

# Import the original pipeline simulator (maintaining sanctity)
from pipeline_simulator import run_pipeline_probe, MAX_CLUES, \
    WORDPLAY_TYPE
from stages.evidence import ComprehensiveWordplayDetector

# Evidence scoring configuration
ENABLE_EVIDENCE_SCORING = True
EVIDENCE_SCORE_WEIGHT = 1.0


class EvidenceAnalyzer:
    """Analyzes successful anagram clues using the permanent engine only."""

    def __init__(self):
        """Initialize the permanent engine detector."""
        self.detector = None
        if ENABLE_EVIDENCE_SCORING:
            try:
                # Load permanent engine with database indicators
                db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
                self.detector = ComprehensiveWordplayDetector(db_path=db_path)
                print("Evidence detector loaded successfully.")
            except Exception as e:
                print(f"WARNING: Evidence detector failed to load: {e}")
                self.detector = None

    def is_successful_anagram_clue(self, record):
        """
        Check if a clue successfully found anagram hits.

        Successful anagram criteria:
        - Has definition candidates
        - Has anagram hits (anagram_hits > 0)
        """
        summary = record.get("summary", {})
        has_def_candidates = summary.get("definition_candidates", 0) > 0
        has_anagram_hits = summary.get("anagram_hits", 0) > 0
        return has_def_candidates and has_anagram_hits

    def apply_evidence_scoring(self, record, debug=False):
        """
        THIN WRAPPER: Calls permanent engine only.
        No embedded logic - everything happens in anagram_evidence_system_patched.py.
        """
        if not self.detector:
            return record

        clue_text = record["clue"]
        candidates = record["definition_candidates"]
        answer = record["answer"]

        # Get definition support for weighting (if available)
        definition_support = record.get("window_support", None)

        if not candidates:
            return record

        # ONLY call to permanent engine - NO embedded logic
        ranking_results = self.detector.analyze_and_rank_anagram_candidates(
            clue_text=clue_text,
            candidates=candidates,
            answer=answer,
            debug=debug,
            definition_support=definition_support
        )

        # Format results for display - preserving exact same output format
        enhanced_record = record.copy()
        enhanced_record["evidence_analysis"] = {
            "evidence_found": ranking_results["evidence_found"],
            "scored_candidates": ranking_results["scored_candidates"][:10],
            # Top 10 for display
            "answer_rank_original": ranking_results["answer_rank_original"],
            "answer_rank_evidence": ranking_results["answer_rank_evidence"],
            "ranking_improved": ranking_results["ranking_improved"]
        }

        return enhanced_record

    def analyze_successful_anagram_cohort(self, results):
        """
        Analyze the successful anagram cohort from pipeline simulator results.
        Returns enhanced results showing how anagrams were successfully solved.
        """
        successful_anagram_clues = [r for r in results if
                                    self.is_successful_anagram_clue(r)]

        print(f"\n[SEARCH] SUCCESSFUL ANAGRAM COHORT ANALYSIS:")
        print(f"Total clues processed: {len(results)}")
        print(f"Successful anagram clues: {len(successful_anagram_clues)}")

        if not successful_anagram_clues:
            print("No successful anagram clues to analyze.")
            return []

        # Count hits by stage
        stage_counts = {
            'anagram_stage': 0,  # solve_type = "anagram_exact"
            'evidence_system': 0  # solve_type starts with "anagram_evidence"
        }

        for record in successful_anagram_clues:
            anagrams = record.get('anagrams', [])
            if anagrams:
                first_hit = anagrams[0]
                solve_type = first_hit.get('solve_type', '')
                if solve_type == 'anagram_exact':
                    stage_counts['anagram_stage'] += 1
                elif solve_type.startswith('anagram_evidence'):
                    stage_counts['evidence_system'] += 1

        print(f"\n[STATS] HITS BY STAGE:")
        print(f"  anagram_stage (brute force):  {stage_counts['anagram_stage']}")
        print(f"  evidence_system (fallback):   {stage_counts['evidence_system']}")

        # Apply evidence scoring to successful anagram clues using permanent engine
        enhanced_results = []
        evidence_improvements = 0
        debug_count = 0  # Limit debug output

        for record in successful_anagram_clues:
            # Add debug flag for first few records
            debug_this = debug_count < 3
            enhanced = self.apply_evidence_scoring(record, debug=debug_this)
            enhanced_results.append(enhanced)
            debug_count += 1

            # Count improvements
            evidence_analysis = enhanced.get("evidence_analysis", {})
            if evidence_analysis.get("ranking_improved", False):
                evidence_improvements += 1

        print(f"Clues with evidence improvements: {evidence_improvements}")

        return enhanced_results


def display_evidence_results(enhanced_results, max_display=400):
    """Display evidence analysis results."""

    # Sort by evidence improvements first, then by evidence found
    # Ensure all values have proper defaults to avoid None comparison errors
    display_results = sorted(enhanced_results,
                             key=lambda r: (
                                 r.get("evidence_analysis", {}).get(
                                     "ranking_improved") or False,
                                 r.get("evidence_analysis", {}).get("evidence_found") or 0
                             ),
                             reverse=True)

    print(f"\n[STATS] EVIDENCE ANALYSIS RESULTS (Top {max_display}):")
    print("=" * 80)

    for i, record in enumerate(display_results[:max_display], 1):
        evidence_analysis = record.get("evidence_analysis", {})

        print(f"\n[{i}] CLUE: {record['clue']}")
        print(f"    TYPE: {record['wordplay_type']}")
        print(f"    ANSWER: {record['answer_raw']}")

        # Show ranking change
        orig_rank = evidence_analysis.get("answer_rank_original")
        evid_rank = evidence_analysis.get("answer_rank_evidence")
        improved = evidence_analysis.get("ranking_improved", False)

        if orig_rank and evid_rank:
            improvement_text = "IMPROVED" if improved else "unchanged"
            print(f"    RANKING: {orig_rank} → {evid_rank} ({improvement_text})")

        # Show evidence found
        evidence_found = evidence_analysis.get("evidence_found", 0)
        print(f"    EVIDENCE: {evidence_found} candidates with evidence")

        # Show top evidence candidates (top 5 unique score levels)
        scored_candidates = evidence_analysis.get("scored_candidates", [])
        if scored_candidates:
            print("    TOP EVIDENCE CANDIDATES:")

            # Extract all unique scores from the full list and sort them
            all_scores = []
            for scored in scored_candidates:
                score = round(scored["evidence_score"], 1)
                if score not in all_scores:
                    all_scores.append(score)

            # Sort scores in descending order and take top 5
            all_scores.sort(reverse=True)
            top_5_scores = all_scores[:5]

            print(
                f"    DEBUG: Found {len(all_scores)} unique scores, showing top 5: {top_5_scores}")

            # Show all candidates with those top 5 score levels (normalized to remove case duplicates)
            display_count = 0
            seen_candidates = set()  # Track normalized candidates to avoid duplicates

            for scored in scored_candidates:
                score = round(scored["evidence_score"], 1)
                if score in top_5_scores:
                    candidate = scored["candidate"]
                    candidate_normalized = candidate.upper().strip()  # Normalize for duplicate checking

                    # Skip if we've already shown this normalized candidate
                    if candidate_normalized in seen_candidates:
                        continue

                    seen_candidates.add(candidate_normalized)
                    display_count += 1
                    evidence = scored["evidence"]

                    marker = "★" if candidate.upper() == record["answer"].upper() else " "
                    evidence_marker = "*" if evidence else "  "

                    print(
                        f"    {marker}{evidence_marker} {display_count:2d}. {candidate:15} (evidence: +{score:.1f})")

                    if evidence:
                        print(
                            f"          → {evidence.evidence_type}: {' + '.join(evidence.fodder_words)}")

                    # Limit total display to avoid overwhelming output
                    if display_count >= 15:
                        remaining = sum(1 for s in scored_candidates
                                        if round(s["evidence_score"], 1) in top_5_scores
                                        and s[
                                            "candidate"].upper().strip() not in seen_candidates)
                        if remaining > 0:
                            print(
                                f"    ... and {remaining} more unique candidates with these score levels")
                        break


def main():
    """Main analysis function."""
    print("[FIX] EVIDENCE-BASED SCORING ANALYSIS")
    print("=" * 60)
    print("Maintaining absolute sanctity of original pipeline simulator")
    print("Analyzing successful anagram cohort (showing how anagrams are solved)")
    print("=" * 60)

    # Initialize evidence analyzer (thin wrapper for permanent engine)
    analyzer = EvidenceAnalyzer()

    # Step 1: Run original pipeline simulator (with appropriate settings for evidence analysis)
    print("\n[LIST] STEP 1: Running original pipeline simulator...")

    # Override the ONLY_MISSING_DEFINITION setting for evidence analysis
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
        raise e

    # Show original results summary
    print("\n[STATS] ORIGINAL PIPELINE RESULTS:")
    print(f"  clues processed           : {overall['clues']}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']}")

    # Step 2: Analyze successful anagram cohort using permanent engine only
    print("\n[SEARCH] STEP 2: Analyzing successful anagram cohort...")
    enhanced_results = analyzer.analyze_successful_anagram_cohort(results)

    # Step 3: Display evidence analysis
    if enhanced_results:
        display_evidence_results(enhanced_results)

    print("\n✅ Analysis complete. Original pipeline simulator untouched.")


if __name__ == "__main__":
    main()