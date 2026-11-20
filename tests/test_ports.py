from iacli.services.ports import ChatInput, ChatPresenter, ConsolePresenter


def test_the_presenter_port_of_a_session_does_not_read_the_input():
    assert "prompt" not in vars(ChatPresenter)


def test_the_console_loop_needs_a_presenter_that_also_reads_the_input():
    assert "prompt" in vars(ChatInput)
    assert {ChatPresenter, ChatInput} <= set(ConsolePresenter.__mro__)
