import subprocess

conf = """[Service]
Environment="HTTP_PROXY=http://172.21.16.1:7897"
Environment="HTTPS_PROXY=http://172.21.16.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
"""

subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "mkdir", "-p", "/etc/systemd/system/docker.service.d"], check=True)
p = subprocess.Popen(["wsl", "-u", "root", "-d", "Ubuntu", "tee", "/etc/systemd/system/docker.service.d/http-proxy.conf"], stdin=subprocess.PIPE, text=True)
p.communicate(input=conf)

subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "systemctl", "daemon-reload"], check=True)
subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "systemctl", "restart", "docker"], check=True)
subprocess.run(["wsl", "-u", "root", "-d", "Ubuntu", "chmod", "666", "/var/run/docker.sock"], check=True)
print("Docker daemon proxy configured and restarted successfully.")
