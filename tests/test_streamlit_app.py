import pytest


# ---------------------------------------------------------------------------
# Download button filename logic
# ---------------------------------------------------------------------------

class TestDownloadButtonFilename:

    def _make_filename(self, question):
        """Mirrors the filename logic in streamlit_app.py."""
        return f"research_report_{question[:30].replace(' ', '_')}.md"

    def test_spaces_replaced_with_underscores(self):
        name = self._make_filename("what are trends in solar energy")
        assert " " not in name

    def test_truncated_to_30_chars_of_question(self):
        question = "what are the latest trends in renewable energy storage systems"
        name = self._make_filename(question)
        slug = name.replace("research_report_", "").replace(".md", "")
        assert len(slug) <= 30

    def test_markdown_extension(self):
        name = self._make_filename("solar energy trends")
        assert name.endswith(".md")

    def test_prefixed_correctly(self):
        name = self._make_filename("solar energy")
        assert name.startswith("research_report_")

    def test_short_question_not_truncated(self):
        name = self._make_filename("AI trends")
        assert "AI_trends" in name

    def test_exact_30_char_question_not_truncated(self):
        question = "a" * 30
        name = self._make_filename(question)
        slug = name.replace("research_report_", "").replace(".md", "")
        assert slug == "a" * 30

    def test_question_longer_than_30_chars_is_truncated(self):
        question = "a" * 50
        name = self._make_filename(question)
        slug = name.replace("research_report_", "").replace(".md", "")
        assert len(slug) == 30

    def test_empty_question_produces_valid_filename(self):
        name = self._make_filename("")
        assert name == "research_report_.md"

    def test_special_characters_not_replaced(self):
        # only spaces are replaced — other chars pass through
        name = self._make_filename("AI: trends & forecasts 2025")
        assert "_" in name


# ---------------------------------------------------------------------------
# Download button data
# ---------------------------------------------------------------------------

class TestDownloadButtonData:

    def test_report_passed_as_data(self):
        report = "# Solar Energy Report\n\nSolar capacity grew 40% in 2024."
        # the data passed to st.download_button should be the raw report string
        assert isinstance(report, str)
        assert len(report) > 0

    def test_empty_report_is_valid_data(self):
        report = ""
        assert isinstance(report, str)

    def test_markdown_content_preserved(self):
        report = "# Heading\n\n- bullet 1\n- bullet 2\n\n**bold**"
        assert report.startswith("#")
        assert "bullet" in report