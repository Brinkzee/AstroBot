import os
import subprocess

proxy_url = os.environ.get("DOCKER_HTTP_PROXY", "http://172.21.16.1:7897")
docker_user = (os.environ.get("WSL_DOCKER_USER") or "").strip()

conf = f"""[Service]
Environment="HTTP_PROXY={proxy_url}"
Environment="HTTPS_PROXY={proxy_url}"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
"""

subprocess.run(
    ["wsl", "-u", "root", "-d", "Ubuntu", "mkdir", "-p", "/etc/systemd/system/docker.service.d"],
    check=True,
)
p = subprocess.Popen(
    ["wsl", "-u", "root", "-d", "Ubuntu", "tee", "/etc/systemd/system/docker.service.d/http-proxy.conf"],
    stdin=subprocess.PIPE,
    text=True,
)
p.communicate(input=conf)
if p.returncode != 0:
    raise subprocess.CalledProcessError(p.returncode, p.args)

subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "systemctl", "daemon-reload"], check=True)
subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "systemctl", "restart", "docker"], check=True)
if docker_user:
    subprocess.run(
        ["wsl", "-u", "root", "-d", "Ubuntu", "usermod", "-aG", "docker", docker_user],
        check=True,
    )
    print(f"Added WSL user '{docker_user}' to the docker group (no insecure socket chmod).")
else:
    print("Tip: grant non-root Docker access safely via `sudo usermod -aG docker $USER` inside WSL.")
print("Docker daemon proxy configured and restarted successfully.")
