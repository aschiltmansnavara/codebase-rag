"""Unit tests for app UI components."""

from unittest.mock import MagicMock, patch

from codebase_rag.app.components import (
    display_chat_history,
    display_sources,
    format_message,
)


class TestDisplaySources:
    """Tests for display_sources function."""

    @patch("codebase_rag.app.components.st")
    def test_display_sources_groups_by_file_path(self, mock_st: MagicMock) -> None:
        """Test that sources with same file path are grouped together."""
        sources = [
            {
                "id": "1",
                "file_path": "data/repos/docs/source/examples/redundancy.ipynb",
                "file_name": "redundancy.ipynb",
            },
            {
                "id": "2",
                "file_path": "data/repos/docs/source/examples/redundancy.ipynb",
                "file_name": "redundancy.ipynb",
            },
        ]

        display_sources(sources)

        # Should render one markdown header + one grouped entry (not two)
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        file_entries = [c for c in markdown_calls if "redundancy.ipynb" in c]
        assert len(file_entries) == 1, f"Expected 1 grouped entry, got {len(file_entries)}"

    @patch("codebase_rag.app.components.st")
    def test_display_sources_empty_list(self, mock_st: MagicMock) -> None:
        """Test display_sources with empty sources list."""
        display_sources([])
        mock_st.markdown.assert_not_called()

    @patch("codebase_rag.app.components.st")
    def test_display_sources_shows_file_path_and_name(self, mock_st: MagicMock) -> None:
        """Test that sources show file path and file name."""
        sources = [
            {
                "id": "1",
                "file_path": "src/main.py",
                "file_name": "[POWER-GRID-MODEL] main.py",
            },
        ]

        display_sources(sources)

        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        entry = [c for c in markdown_calls if "main.py" in c]
        assert len(entry) == 1
        assert "src/main.py" in entry[0]
        assert "[POWER-GRID-MODEL]" in entry[0]


class TestFormatMessage:
    """Tests for format_message function."""

    @patch("codebase_rag.app.components.display_sources")
    @patch("codebase_rag.app.components.st")
    def test_format_message_with_sources(self, mock_st: MagicMock, mock_display_sources: MagicMock) -> None:
        """Test formatting message with sources."""
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()

        message = {
            "role": "assistant",
            "content": "Test response",
            "sources": [{"id": "1", "file_path": "test.ipynb", "file_name": "test.ipynb"}],
        }

        format_message(message)

        mock_display_sources.assert_called_once_with(message["sources"])

    @patch("codebase_rag.app.components.display_sources")
    @patch("codebase_rag.app.components.st")
    def test_format_message_without_sources(self, mock_st: MagicMock, mock_display_sources: MagicMock) -> None:
        """Test formatting message without sources."""
        message = {"role": "user", "content": "Test question"}
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()

        format_message(message)

        mock_display_sources.assert_not_called()

    @patch("codebase_rag.app.components.display_sources")
    @patch("codebase_rag.app.components.st")
    def test_format_message_default_index(self, mock_st: MagicMock, mock_display_sources: MagicMock) -> None:
        """Test formatting message with default index and empty sources."""
        message = {"role": "assistant", "content": "Test", "sources": []}
        mock_st.chat_message.return_value.__enter__ = MagicMock()
        mock_st.chat_message.return_value.__exit__ = MagicMock()

        format_message(message)

        mock_display_sources.assert_not_called()


class TestDisplayChatHistory:
    """Tests for display_chat_history function."""

    @patch("codebase_rag.app.components.format_message")
    @patch("codebase_rag.app.components.st")
    def test_display_chat_history_with_multiple_messages(self, mock_st: MagicMock, mock_format: MagicMock) -> None:
        """Test displaying chat history with multiple messages."""
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1", "sources": []},
            {"role": "user", "content": "Question 2"},
        ]
        mock_st.session_state.messages = messages

        display_chat_history()

        if mock_format.call_count != 3:
            raise AssertionError(f"Expected 3 format_message calls, got {mock_format.call_count}")
        mock_format.assert_any_call(messages[0])
        mock_format.assert_any_call(messages[1])
        mock_format.assert_any_call(messages[2])

    @patch("codebase_rag.app.components.format_message")
    @patch("codebase_rag.app.components.st")
    def test_display_chat_history_empty(self, mock_st: MagicMock, mock_format: MagicMock) -> None:
        """Test displaying empty chat history."""
        mock_st.session_state.messages = []

        display_chat_history()

        mock_format.assert_not_called()
