#!/usr/bin/env bash
# Explicit privileged bootstrap for Debian/Ubuntu hosts only.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run with sudo: sudo ./scripts/install_postgres_debian.sh" >&2
  exit 1
fi

apt-get update
apt-get install -y postgresql postgresql-client
systemctl enable --now postgresql

echo "PostgreSQL is running. Next run scripts/bootstrap_postgres.py with admin credentials."
