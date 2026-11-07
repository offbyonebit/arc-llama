# Isolated demo environment: source this before any demo step. Nothing here reads
# the real Arc Llama config, OpenCode data or shell history.
export DEMO=${DEMO:-/tmp/arc-demo}
export HOME=$DEMO/home XDG_CONFIG_HOME=$DEMO/config XDG_STATE_HOME=$DEMO/state \
       XDG_DATA_HOME=$DEMO/data XDG_CACHE_HOME=$DEMO/cache HERMES_HOME=$DEMO/hermes
export BROWSER=$DEMO/bin/open-url   # the stub records the URL; Playwright opens it
export ARC_LLAMA_EXPERIMENTAL_AGENT=1   # Work is still an experimental feature in 1.1
unset ARC_LLAMA_ADMIN_TOKEN ARC_LLAMA_OPENCODE_BIN ARC_LLAMA_HERMES_BIN
export PATH=$DEMO/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$DEMO/bin" "$DEMO/models"
