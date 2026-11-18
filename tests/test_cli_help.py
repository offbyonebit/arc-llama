import click

from arc_llama.cli import cli


def test_all_cli_commands_and_visible_options_have_help():
    missing = []

    def walk(command, path):
        if not (command.help or "").strip():
            missing.append(f"{path}: missing command help")
        for param in command.params:
            if isinstance(param, click.Option) and not param.hidden and "--help" not in param.opts:
                if not (param.help or "").strip():
                    missing.append(f"{path} {'/'.join(param.opts)}: missing option help")
        if isinstance(command, click.Group):
            for name, child in command.commands.items():
                walk(child, f"{path} {name}")

    walk(cli, "arc-llama")
    assert not missing, "\n".join(missing)
