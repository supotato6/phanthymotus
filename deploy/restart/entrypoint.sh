#!/bin/sh
# entrypoint.sh — extract compose from new image, replace image ref, restart service
# Aligned with install.sh behavior: full compose extraction + sed replacement + clean restart
# Env: COMPOSE_DIR, SERVICE, NEW_IMAGE, CONTAINER_NAME (optional)
set -e

: "${COMPOSE_DIR:?COMPOSE_DIR required}"
: "${SERVICE:?SERVICE required}"
: "${NEW_IMAGE:?NEW_IMAGE required}"

CONTAINER_NAME="${CONTAINER_NAME:-phanthy-motus-${SERVICE}-1}"

echo "[restart] service=${SERVICE} image=${NEW_IMAGE}"
echo "[restart] compose_dir=${COMPOSE_DIR}"
echo "[restart] container=${CONTAINER_NAME}"

COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"

# ── Step 1: Extract compose from new image (single source of truth) ──────────
echo "[restart] extracting compose from new image..."
CID=$(docker create "${NEW_IMAGE}")
docker cp "${CID}:/deploy/docker-compose.yml" /tmp/new-compose.yml 2>/dev/null || true
docker rm "${CID}" >/dev/null

if [ ! -f /tmp/new-compose.yml ]; then
    echo "[restart] ERROR: /deploy/docker-compose.yml not found in image"
    exit 1
fi

# ── Step 2: Update compose file ──────────────────────────────────────────────
# Strategy: if other services exist in current compose, preserve them and only
# replace the target service definition. Otherwise use extracted file directly.

if [ -f "${COMPOSE_FILE}" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "${COMPOSE_FILE}" /tmp/new-compose.yml "${NEW_IMAGE}" "${SERVICE}" <<'PY'
import sys, yaml

compose_path, new_path, new_image, service = sys.argv[1:5]

with open(compose_path) as f:
    existing = yaml.safe_load(f) or {}

with open(new_path) as f:
    new = yaml.safe_load(f) or {}

existing_services = existing.get('services', {})
new_services = new.get('services', {})

# Check if there are other services beyond the target one
other_services = {k: v for k, v in existing_services.items() if k != service}

if other_services and service in new_services:
    # Preserve other services, replace target service with new definition
    new_services[service]['image'] = new_image
    existing_services[service] = new_services[service]
    existing['services'] = existing_services
    with open(compose_path, 'w') as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
else:
    # No other services — use extracted compose directly (same as install.sh)
    import shutil
    shutil.copy2(new_path, compose_path)
    # Replace __IMAGE__ placeholder and any existing image line for the service
    with open(compose_path) as f:
        content = f.read()
    content = content.replace('__IMAGE__', new_image)
    with open(compose_path, 'w') as f:
        f.write(content)
PY
else
    # No existing compose or no python3 — use extracted file with sed (same as install.sh)
    cp /tmp/new-compose.yml "${COMPOSE_FILE}"
    sed -i "s|image:.*__IMAGE__.*|image: ${NEW_IMAGE}|" "${COMPOSE_FILE}"
    # Also replace any existing image line under the service block
    sed -i "/^  ${SERVICE}:/,/^  [^ ]/{s|image:.*|image: ${NEW_IMAGE}|}" "${COMPOSE_FILE}"
fi

echo "[restart] compose file updated"

# ── Step 3: Stop and remove old container (clean slate, same as install.sh) ──
echo "[restart] stopping old container..."
cd "${COMPOSE_DIR}"
docker compose stop "${SERVICE}" 2>/dev/null || true
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── Step 4: Start service ────────────────────────────────────────────────────
echo "[restart] starting service..."
docker compose up -d "${SERVICE}"

echo "[restart] done. ${SERVICE} is up."
