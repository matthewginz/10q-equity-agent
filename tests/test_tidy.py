import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.tidy import tidy_output


def test_drops_the_stance_heading_the_badge_already_shows():
    text = "### Final Equity Research Stance: Bullish\n\n**Sentiment**\nPositive."
    assert tidy_output(text) == "**Sentiment**\nPositive."


def test_drops_title_block_and_filing_metadata():
    text = ("**Equity Research Snapshot: NXP Semiconductors N.V. (NXPI)**\n**Date:** April 28, 2026\n"
            "**Period Ending:** March 29, 2026\n\n### **Profitability Trend**\nGross margin of 56.21%.")
    assert tidy_output(text) == "### **Profitability Trend**\nGross margin of 56.21%."


def test_drops_pure_lead_in_sentences():
    text = ("Based on the provided 10-Q filing for the period ended March 29, 2026, here is the extracted "
            "information:\n\n**1. Segment Breakdown**\nConsolidated only.")
    assert tidy_output(text) == "**1. Segment Breakdown**\nConsolidated only."
    text = ("### **Cross-Check Analysis: Quantitative vs. Qualitative Synthesis**\n\nUpon reviewing the "
            "quantitative metrics (Step 1) against the narrative, the following alignment and discrepancies "
            "are noted:\n\n**1. Operational Leverage**\nRevenue up 12%.")
    assert tidy_output(text) == "**1. Operational Leverage**\nRevenue up 12%."


def test_keeps_an_opening_that_carries_a_finding():
    text = ("Based on the provided excerpt from Sysco Corp's 10-Q filing, there is insufficient information "
            "to identify or rank specific risks.\n\nThe text contains only administrative sections.")
    assert tidy_output(text) == text


def test_keeps_a_real_first_section_heading():
    text = "**1. Segment or geographic revenue breakdown**\nNot reported."
    assert tidy_output(text) == text


def test_shortens_stance_section_titles():
    text = ("### Final Equity Research Stance: Neutral\n\n**1. Sentiment**\nMixed.\n\n"
            "**3. The single biggest supporting factor and the single biggest risk to the thesis**\nCash.\n\n"
            "**4. Analyst Recommendation**\nHold.")
    assert tidy_output(text) == ("**Sentiment**\nMixed.\n\n**Biggest support and biggest risk**\nCash.\n\n"
                                 "**Recommendation**\nHold.")


def test_drops_date_of_report_style_metadata():
    text = "**Date of Report:** 2024-10-31\n\n#### Profitability Trend\nEPS of $1.42."
    assert tidy_output(text) == "#### Profitability Trend\nEPS of $1.42."


def test_drops_only_the_process_sentence_and_keeps_the_finding_after_it():
    text = ("As an equity research analyst, I have cross-referenced the consolidated metrics (Step 1) against "
            "the segment disclosures (Steps 2 & 3). There are several notable tensions in the data.\n\n**1. Margins**")
    assert tidy_output(text) == "There are several notable tensions in the data.\n\n**1. Margins**"
    text = ("As an equity research analyst, I have cross-referenced the data (Step 1) against the risks (Step 2). "
            "Below is the assessment of the alignment between the narrative and the numbers.\n\n**1. Margins**")
    assert tidy_output(text) == "**1. Margins**"


def test_keeps_a_cross_check_opening_that_states_a_conclusion():
    text = ("Based on a cross-check of the quantitative data (Step 1) against the segment performance (Step 3), "
            "there is a notable divergence between the outlook and the segment data.\n\n**1. Margins**")
    assert tidy_output(text) == text


def test_drops_a_short_opening_that_just_leads_into_a_colon():
    text = "Based on the provided 10-Q filing for the period ended March 31, 2026:\n\n1. **Segments:** none."
    assert tidy_output(text) == "1. **Segments:** none."


def test_never_returns_empty():
    text = "### Final Equity Research Stance: Bullish"
    assert tidy_output(text) == text
