# Jupyter Server configuration — ensures notebooks always open at /app
# regardless of how Jupyter is launched (docker compose, VS Code, etc.).
c.ServerApp.root_dir = "/app"
c.ServerApp.preferred_dir = "/app"
