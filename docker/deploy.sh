#!/usr/bin/env bash
# Deploys Lodestar to an Ubuntu server over SSH: installs Docker, ships the image, writes the server's .env,
# optionally copies the local database, and starts every service behind Caddy (HTTPS).
#
#   docker build -f docker/Dockerfile -t lodestar:local .         # build first (same CPU type as the server)
#   docker/deploy.sh <server-ip> <ssh-key.pem> [--with-data]
#
# The site is served at https://<ip-with-dashes>.sslip.io unless DOMAIN is set in the environment.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,8p' "$0"
  exit 1
fi

IP="$1"
KEY="$2"
WITH_DATA="${3:-}"
IMAGE="lodestar:local"
REMOTE="ubuntu@${IP}"
DOMAIN="${DOMAIN:-${IP//./-}.sslip.io}"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
cd "$(dirname "$0")/.."

remote() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
step() { printf '\n==> %s\n' "$*"; }

chmod 400 "$KEY"
[[ -f .env ]] || { echo ".env not found: the server's settings are built from it" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null || { echo "Image $IMAGE not found; build it first" >&2; exit 1; }

step "Checking the server"
server_arch="$(remote uname -m)"
image_arch="$(docker image inspect "$IMAGE" --format '{{.Architecture}}')"
case "$server_arch/$image_arch" in
  aarch64/arm64 | x86_64/amd64) echo "server $server_arch, image $image_arch: OK" ;;
  *) echo "Server is $server_arch but the image is $image_arch. Use a matching instance type, or build with --platform." >&2; exit 1 ;;
esac

step "Installing Docker and a 2 GB swap file (skipped if already there)"
remote 'set -e
  if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sudo sh; sudo usermod -aG docker "$USER"; fi
  if ! sudo swapon --show | grep -q /swapfile; then
    sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
  mkdir -p ~/lodestar/docker'

step "Copying the image ($(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.0f MB", $1/1e6}') before compression)"
docker save "$IMAGE" | gzip -1 | remote 'gunzip | sudo docker load'

step "Copying compose.yaml, the Caddyfile and the server's .env"
scp "${SSH_OPTS[@]}" compose.yaml "$REMOTE:lodestar/compose.yaml"
scp "${SSH_OPTS[@]}" docker/Caddyfile "$REMOTE:lodestar/docker/Caddyfile"
admin_token="$(grep -E '^ADMIN_TOKEN=.+' .env | cut -d= -f2- || true)"
admin_token="${admin_token:-$(openssl rand -hex 24)}"
{
  grep -vE '^(COMPOSE_PROFILES|DOMAIN|PUBLIC_BASE_URL|LODESTAR_IMAGE|ADMIN_TOKEN)=' .env
  echo
  echo "# Written by docker/deploy.sh"
  echo "COMPOSE_PROFILES=app,proxy"
  echo "DOMAIN=${DOMAIN}"
  echo "PUBLIC_BASE_URL=https://${DOMAIN}"
  echo "LODESTAR_IMAGE=${IMAGE}"
  echo "ADMIN_TOKEN=${admin_token}"
} | remote 'umask 077 && cat > ~/lodestar/.env'

if [[ "$WITH_DATA" == "--with-data" ]]; then
  step "Copying the local database (documents, chunks, embeddings, digests, subscribers)"
  remote 'cd ~/lodestar && sudo docker compose up -d --wait db'
  docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
    | remote 'cd ~/lodestar && sudo docker compose exec -T db sh -c "pg_restore -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" --clean --if-exists --no-owner"' \
    || echo "pg_restore reported warnings (usually harmless on a fresh database); continuing"
fi

step "Starting Lodestar (first start downloads the models; this can take a few minutes)"
remote 'cd ~/lodestar && sudo docker compose up -d --wait'
remote 'cd ~/lodestar && sudo docker compose ps --format "{{.Service}}\t{{.Status}}"'

step "Done"
echo "Dashboard: https://${DOMAIN}"
echo "Admin token (for POST /admin/ingest) is in ~/lodestar/.env on the server."
