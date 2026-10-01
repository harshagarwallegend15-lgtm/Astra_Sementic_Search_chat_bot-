"""Tests for the deterministic grounding verifier.

These are the tests that protect the honesty guarantees, so they deliberately
cover fabricated numbers, invented years, unknown markers and honest refusals.
"""

from __future__ import annotations

import pytest

from src.models import Chunk, RetrievedChunk

from src.grounding import (
    ISSUE_MISSING_MARKER,
    ISSUE_SOURCE_ECHO,
    ISSUE_UNKNOWN_MARKER,
    ISSUE_UNSUPPORTED_NUMBER,
    build_context_text,
    detect_refusal,
    extract_markers,
    find_source_echo,
    find_unsupported_numbers,
    normalise_markers,
    strip_all_markers,
    strip_trailing_hedge,
    verify,
)
from src.grounding import MARKER_PATTERN


class TestMarkerExtraction:
    def test_markers_are_ordered_and_unique(self):
        assert extract_markers("a [S2] b [S1] c [S2]") == ["S2", "S1"]

    def test_lowercase_markers_normalise(self):
        assert extract_markers("see [s3]") == ["S3"]

    def test_no_markers(self):
        assert extract_markers("plain text") == []

    def test_compound_marker_reduced_to_first_source(self):
        assert normalise_markers("claim [S1, p. 14]") == "claim [S1]"

    def test_multi_source_bracket_reduced(self):
        assert normalise_markers("claim [S1, S2]") == "claim [S1]"

    def test_strip_all_markers_removes_every_marker(self):
        assert strip_all_markers("a [S1] b [S2]") == "a  b"

    def test_unicode_bracket_markers_fold_to_ascii(self):
        # Groq's gpt-oss-120b emits lenticular brackets rather than ASCII, which
        # would otherwise look like unrecognised text and silently drop citations.
        assert normalise_markers("capabilities\u3010S1\u3011 and more\u300aS3\u300b") == (
            "capabilities[S1] and more[S3]"
        )

    def test_unicode_bracket_variants_are_all_parsed(self):
        for raw in ("\u3010S1\u3011", "\u3014S1\u3015", "\u27e8S1\u27e9", "\uff08S1\uff09", "\uff5bS1\uff5d"):
            assert extract_markers(normalise_markers(f"claim {raw}")) == ["S1"], raw

    def test_fullwidth_digits_inside_marker_are_parsed(self):
        assert extract_markers(normalise_markers("claim \u3010S\uff15\uff11\u3011")) == ["S51"]

    def test_marker_with_interior_whitespace_is_canonicalised(self):
        # gpt-oss writes "[ S1 ]" for bullet lists. MARKER_PATTERN alone only
        # tolerates one optional space, so the citation silently vanished and the
        # fully written answer was scored as no_context.
        assert normalise_markers("list [ S1 ].") == "list [S1]."
        assert extract_markers(normalise_markers("a [S 2] b")) == ["S2"]
        assert extract_markers(normalise_markers("a [ s3 ] b")) == ["S3"]

    def test_spaced_compound_markers_still_collapse(self):
        assert normalise_markers("cite [ S1, S2 ].") == "cite [S1]."
        assert normalise_markers("cite [ S1, p. 2 ].") == "cite [S1]."
        assert normalise_markers("cite [S 1, S2 ].") == "cite [S1]."

    def test_spaced_bare_reference_still_dropped(self):
        assert normalise_markers("ref [ 59 ] here [S5]") == "ref  here [S5]"


class TestNumericVerification:
    def test_fabricated_percentage_is_detected(self, uav_chunk):
        answer = "75% of UAV missions are fully autonomous [S1]."
        report = verify(answer, [uav_chunk])
        assert "75" in report.unsupported_numbers
        assert not report.is_grounded
        assert ISSUE_UNSUPPORTED_NUMBER in {issue.kind for issue in report.issues}

    def test_fabricated_fraction_is_detected(self, uav_chunk):
        answer = "About 62.5 percent of missions are autonomous."
        assert "62.5" in find_unsupported_numbers(answer, build_context_text([uav_chunk]))

    def test_invented_year_is_detected(self, uav_chunk):
        report = verify("Guidance from 2019 mandates this [S1].", [uav_chunk])
        assert "2019" in report.unsupported_numbers

    def test_number_present_in_context_is_allowed(self, uav_chunk):
        text = "The guidance document referenced is from 2019 and covers this."
        chunk = RetrievedChunk(
            chunk=Chunk(
                chunk_id="c", document_id="d", filename="f.pdf",
                title="Unmanned aerial vehicle", page=3, text=text,
                section="History",
            ),
            score=0.6,
        )
        report = verify("The referenced guidance is from 2019 [S1].", [chunk])
        assert report.unsupported_numbers == []
        assert report.is_grounded

    def test_three_is_allowed_when_context_says_three(self, ew_chunk):
        answer = "Electronic warfare has three major subdivisions [S1]."
        report = verify(answer, [ew_chunk])
        assert report.unsupported_numbers == []
        assert report.is_grounded

    def test_number_word_absent_from_context_is_detected(self, uav_chunk):
        report = verify("There are three levels of autonomy [S1].", [uav_chunk])
        assert "3" in report.unsupported_numbers

    def test_citation_index_is_not_treated_as_a_claim(self, mixed_chunks):
        answer = "Autonomy varies widely [S1] and the OODA loop quantifies it [S3]."
        report = verify(answer, mixed_chunks)
        assert report.unsupported_numbers == []

    def test_currency_symbol_does_not_hide_a_number(self, uav_chunk):
        found = find_unsupported_numbers("It costs $5 million.", build_context_text([uav_chunk]))
        assert "5" in found


class TestMarkerValidation:
    def test_unknown_marker_is_flagged(self, mixed_chunks):
        report = verify("Autonomy varies widely [S1] and more [S9].", mixed_chunks)
        assert report.unknown_markers == ["S9"]
        assert not report.is_grounded
        assert ISSUE_UNKNOWN_MARKER in {issue.kind for issue in report.issues}

    def test_all_markers_resolve(self, mixed_chunks):
        report = verify("All three documents say something [S1][S2][S3].", mixed_chunks)
        assert report.valid_markers == ["S1", "S2", "S3"]
        assert report.is_grounded

    def test_missing_citations_flagged_when_required(self, uav_chunk):
        report = verify("Autonomy varies widely across manufacturers.", [uav_chunk])
        assert ISSUE_MISSING_MARKER in {issue.kind for issue in report.issues}

    def test_refusal_needs_no_citations(self, uav_chunk):
        report = verify("The documents do not state this.", [uav_chunk])
        assert report.is_refusal
        assert ISSUE_MISSING_MARKER not in {issue.kind for issue in report.issues}

    def test_require_citations_can_be_disabled(self, uav_chunk):
        report = verify("Autonomy varies widely.", [uav_chunk], require_citations=False)
        assert report.issues == []


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "The supplied documents do not state this.",
            "The documents do not mention that percentage.",
            "This is not covered by the provided material.",
            "No information about that is given.",
        ],
    )
    def test_detects_refusals(self, text):
        assert detect_refusal(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Autonomy varies widely across manufacturers.",
            "Electronic warfare has three subdivisions.",
        ],
    )
    def test_does_not_flag_normal_answers(self, text):
        assert not detect_refusal(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Answers first, then hedges. This is a good answer, not a refusal.
            (
                "Both are unmanned robots used in war and by civilians. [S1] "
                "UAVs operate in the air and UGVs on land. [S1] UGVs are "
                "considered Remote-Operated or Autonomous. [S4] The documents "
                "do not provide an explicit side-by-side comparison table."
            ),
            (
                "One advantage is that endurance is not constrained by the "
                "physiological limits of a pilot [S2]. Another is solar flight "
                "[S8]. The supplied documents do not state any other advantages."
            ),
        ],
    )
    def test_a_trailing_caveat_is_not_a_refusal(self, text):
        """A caveat after cited content must not downgrade a good answer."""
        assert not detect_refusal(text)

    def test_a_refusal_that_cites_related_content_is_still_a_refusal(self):
        assert detect_refusal(
            "The documents do not state this. [S2] covers the related concept "
            "of autonomy degrees."
        )

    def test_honest_refusal_with_cited_context_is_grounded(self, uav_chunk):
        answer = (
            "The supplied documents do not state what percentage of UAV missions "
            "are fully autonomous. They describe that autonomy varies widely and "
            "that full autonomy is available for specific tasks such as airborne "
            "refuelling [S1]."
        )
        report = verify(answer, [uav_chunk])
        assert report.is_refusal
        assert report.is_grounded
        assert report.unsupported_numbers == []
        assert report.valid_markers == ["S1"]

    def test_refusal_still_blocks_fabricated_numbers(self, uav_chunk):
        answer = "The documents do not state it, but it is roughly 40% [S1]."
        report = verify(answer, [uav_chunk])
        assert not report.is_grounded
        assert report.unsupported_numbers == []  # reported separately, not blocking twice


class TestNormaliseMarkers:
    def test_compact_page_marker_keeps_the_source(self):
        assert normalise_markers("Attack and protection [S1, p. 2].") == (
            "Attack and protection [S1]."
        )

    def test_multiple_sources_in_one_marker(self):
        assert normalise_markers("Both [S1, p. 2][S2].") == "Both [S1][S2]."

    def test_plain_marker_is_untouched(self):
        assert normalise_markers("Grounded [S1].") == "Grounded [S1]."

    def test_bare_wikipedia_reference_markers_are_dropped(self):
        """The source PDFs carry their own "[59]" bibliography markers."""
        assert normalise_markers("Taifun-M was unveiled [59][60] [S5].") == (
            "Taifun-M was unveiled  [S5]."
        )

    def test_a_real_citation_is_never_swallowed(self):
        for text in ("Answer [S1].", "[S12]", "[S1][S2][S3]"):
            assert normalise_markers(text) == text

    def test_numbered_lists_are_left_alone(self):
        assert normalise_markers("1. First [S1]\n2. Second") == (
            "1. First [S1]\n2. Second"
        )

    def test_reference_ranges_are_dropped(self):
        assert normalise_markers("cited [12-15] here [S1]") == "cited  here [S1]"


class TestSourceEchoDetection:
    def test_detects_a_copied_source_block(self, uav_chunk):
        answer = (
            "[S1] Document: Unmanned aerial vehicle\n"
            "Page: 14\n"
            "Section: Autonomy\n"
            "Text: The level of autonomy in UAVs varies widely."
        )
        report = verify(answer, [uav_chunk])
        assert report.source_echo is not None

    def test_normal_answer_is_not_an_echo(self, uav_chunk):
        report = verify("Autonomy varies widely between models [S1].", [uav_chunk])
        assert report.source_echo is None

    def test_a_single_ordinary_word_is_not_an_echo(self, uav_chunk):
        report = verify("The text describes a text-to-speech system [S1].", [uav_chunk])
        assert report.source_echo is None

    def test_echo_is_recorded_as_an_issue(self, uav_chunk):
        answer = "Document: X Page: 1 Section: Y Text: autonomy [S1]"
        report = verify(answer, [uav_chunk])
        assert ISSUE_SOURCE_ECHO in {issue.kind for issue in report.issues}


class TestHonestyScenarios:
    def test_q10_fabrication_is_rejected(self, uav_chunk):
        """The core honesty test: no autonomy percentage exists in the corpus."""
        answer = (
            "According to the documents, 80% of UAV missions are flown fully "
            "autonomously [S1]."
        )
        report = verify(answer, [uav_chunk])
        assert not report.is_grounded
        assert report.unsupported_numbers == ["80"]

    def test_q10_partial_answer_is_accepted(self, uav_chunk):
        answer = (
            "The documents do not give a percentage for fully autonomous missions. "
            "They state that the level of autonomy in UAVs varies widely and that "
            "full autonomy is available for specific tasks such as airborne "
            "refuelling [S1]."
        )
        report = verify(answer, [uav_chunk])
        assert report.is_grounded
        assert report.is_refusal

    def test_q2_ground_truth_is_accepted(self, ew_chunk):
        answer = (
            "Electronic warfare consists of three subdivisions: electronic attack, "
            "electronic protection, and electronic warfare support [S1]."
        )
        report = verify(answer, [ew_chunk])
        assert report.is_grounded
        assert report.valid_markers == ["S1"]

    def test_invented_section_name_is_a_limit_of_the_numeric_gate(self, ew_chunk):
        """Documents the boundary of the deterministic verifier.

        "Mission types" is a section that does not exist in the EW document, but
        it carries no invented number, so the numeric gate passes it. A bare
        invented heading is caught by the system prompt and by the model's own
        instruction-following, not by :func:`verify`. This test exists so the
        limitation is explicit rather than assumed away.
        """
        answer = "The Mission types section lists three categories [S1]."
        report = verify(answer, [ew_chunk])
        assert report.unsupported_numbers == []


class TestTrailingHedgeRemoval:
    """A disclaimer bolted onto a finished answer reads as the system
    undermining itself. Prompting against it is unreliable, so the sentence is
    removed deterministically. These tests pin both halves: strip the noise,
    never touch an honest refusal."""

    ANSWER = (
        "UAV endurance is not constrained by the physiological capabilities of a "
        "human pilot [S2]. Solar power can extend flight times at altitude [S8]."
    )

    def test_trailing_disclaimer_is_removed(self):
        text = (
            self.ANSWER
            + "\n\nThe supplied documents do not state any other advantages of "
            "long-endurance UAVs over manned platforms."
        )
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_note_that_disclaimer_is_removed(self):
        text = (
            self.ANSWER
            + "\n\nNote that the sources do not provide a direct comparison "
            "between UAVs and UGVs."
        )
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_several_disclaimer_paragraphs_are_removed(self):
        text = (
            self.ANSWER
            + "\n\nThe documents do not discuss battery technology."
            + "\n\nHowever, the sources do not specify endurance figures."
        )
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_single_newline_separation_is_handled(self):
        text = self.ANSWER + "\nThe supplied documents do not state any other advantages."
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_genuine_refusal_is_preserved(self):
        """Nothing but the disclaimer: this is an honest refusal, not noise."""
        text = (
            "The supplied documents do not state the percentage of UAV missions "
            "that are fully autonomous."
        )
        assert strip_trailing_hedge(text) == text

    def test_refusal_with_cited_context_is_preserved(self):
        text = (
            "The supplied documents do not state the percentage of UAV missions "
            "that are fully autonomous, though they describe levels of autonomy [S1]."
        )
        assert strip_trailing_hedge(text) == text

    def test_cited_disclaimer_paragraph_is_preserved(self):
        """A disclaimer carrying its own marker is making a sourced claim."""
        text = (
            self.ANSWER
            + "\n\nThe documents do not discuss battery technology, and no "
            "citation is available for it [S9]."
        )
        assert strip_trailing_hedge(text) == text

    def test_uncited_remainder_is_never_produced(self):
        """Stripping must not leave an answer with no citations at all."""
        text = "Short bit [S1].\n\nThe documents do not state anything else."
        result = strip_trailing_hedge(text)
        assert MARKER_PATTERN.search(result) or result == "Short bit [S1]."

    def test_a_real_answer_ending_in_prose_is_untouched(self):
        text = self.ANSWER
        assert strip_trailing_hedge(text) == text

    def test_empty_input_is_safe(self):
        assert strip_trailing_hedge("") == ""
        assert strip_trailing_hedge("   ") == "   "

    def test_mid_answer_disclaimer_is_not_removed(self):
        """Only the tail is a disclaimer; this one separates real content."""
        text = (
            "The documents do not state fuel figures [S1].\n\n"
            "Endurance instead depends on airframe and payload [S2].\n\n"
            "Solar panels extend flight time [S8]."
        )
        assert strip_trailing_hedge(text) == text


class TestTrailingHedgeShapes:
    """The shapes models actually produce, captured from real hosted-Llama
    answers during development. Each of these shipped in a graded `grounded`
    response and read as the system arguing with itself."""

    ANSWER = (
        "UAVs have civilian, commercial, military, and aerospace "
        "applications, and they are used for surveillance and targeting [S7]."
    )

    def test_hedge_after_a_citation_marker_is_removed(self):
        """Observed on Q1. The marker between the sentences defeats a naive
        sentence split, because the split looks for sentence punctuation."""
        text = (
            self.ANSWER
            + " However, this source does not provide information on the main "
            "categories or classes of UAVs."
        )
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_inline_hedge_in_a_single_paragraph_is_removed(self):
        """A one-paragraph answer has no blank line to split on."""
        text = (
            "Jamming happened during the Russo-Japanese War of 1904-1905. [S3] [S4] "
            "However, it does not provide a clear distinction between the two events."
        )
        result = strip_trailing_hedge(text)
        assert "does not provide a clear distinction" not in result
        assert "[S3]" in result

    def test_whole_final_paragraph_that_is_one_hedge_sentence_is_removed(self):
        """Observed on Q6: the disclaimer is the entire last paragraph and names
        the documents only as a pronoun."""
        text = (
            "Jamming happened during the Russo-Japanese War of 1904-1905. [S3] [S4]"
            "\n\nHowever, it does not provide a clear distinction between the two "
            "events or a definition of what constitutes jamming."
        )
        result = strip_trailing_hedge(text)
        assert "does not provide a clear distinction" not in result
        assert "[S3]" in result

    def test_adverb_between_not_and_verb_is_recognised(self):
        """Observed on Q5 retry: 'do not *explicitly* state'."""
        text = (
            self.ANSWER
            + "\n\nNote that the sources do not explicitly state the advantages "
            "of long-endurance UAVs over manned platforms, but rather highlight "
            "the benefits of UAVs in general."
        )
        assert strip_trailing_hedge(text) == self.ANSWER

    def test_redundant_trailing_restatement_of_a_refusal_is_trimmed(self):
        """Observed on Q10. The refusal stands on its own; repeating it at the
        end is padding, and removing it must not soften the honesty claim."""
        text = (
            "The supplied documents do not state the percentage of UAV missions "
            "that are fully autonomous, though they describe levels of autonomy "
            "[S1]. UAVs can be optionally piloted [S2]. However, they do not "
            "provide information on the percentage of fully autonomous UAV missions."
        )
        result = strip_trailing_hedge(text)
        assert "However, they do not provide" not in result
        assert "do not state the percentage" in result
        assert "[S1]" in result and "[S2]" in result
