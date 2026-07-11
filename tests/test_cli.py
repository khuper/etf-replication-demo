import unittest

from src.cli import HELP, SHELL_EXAMPLE, build_parser


class CliHelpTests(unittest.TestCase):
    def test_shell_help_includes_copyable_example(self):
        self.assertIn(SHELL_EXAMPLE, build_parser().format_help())

    def test_interactive_help_includes_shell_and_prompt_examples(self):
        self.assertIn("target PSP", HELP)
        self.assertIn(SHELL_EXAMPLE, HELP)


if __name__ == "__main__":
    unittest.main()
