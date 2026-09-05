"""Smallest useful arc-llama plugin example."""

from fastapi import FastAPI

from arc_llama.plugins import Plugin


class HelloPlugin(Plugin):
    name = "hello"

    def register(self, app: FastAPI) -> None:
        @app.get("/plugin/hello")
        async def hello() -> dict[str, str]:
            return {"plugin": self.name, "message": "hello from an arc-llama plugin"}

    def startup(self, app: FastAPI) -> None:
        app.state.hello_plugin_started = True

    def shutdown(self, app: FastAPI) -> None:
        app.state.hello_plugin_stopped = True


def create_plugin() -> HelloPlugin:
    return HelloPlugin()
