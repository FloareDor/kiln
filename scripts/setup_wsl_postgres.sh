#!/usr/bin/env bash
# Installs and starts PostgreSQL inside WSL2 for local development and tests.
#
# Native rather than Docker: Docker Desktop's daemon is not always running on this
# box, and the store tests should not fail for a reason that has nothing to do with
# the code. Kubernetes runs Postgres as a StatefulSet; this is only the dev database.
set -euo pipefail

DB_NAME="${KILN_DB:-kiln}"
DB_USER="${KILN_DB_USER:-kiln}"
DB_PASS="${KILN_DB_PASS:-kiln}"

if ! command -v psql >/dev/null 2>&1; then
  echo "==> installing postgresql"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq postgresql postgresql-contrib >/dev/null
fi

# WSL has no systemd by default, so the cluster is driven through pg_ctlcluster.
VER=$(ls /etc/postgresql | sort -V | tail -1)
if ! pg_isready -q 2>/dev/null; then
  echo "==> starting cluster $VER"
  pg_ctlcluster "$VER" main start || service postgresql start
fi

for _ in $(seq 1 30); do
  pg_isready -q && break
  sleep 1
done
pg_isready || { echo "postgres did not come up" >&2; exit 1; }

echo "==> ensuring role and database"
su - postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'\"" | grep -q 1 || \
  su - postgres -c "psql -q -c \"CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}' SUPERUSER\""

su - postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'\"" | grep -q 1 || \
  su - postgres -c "createdb -O ${DB_USER} ${DB_NAME}"

# A separate database for the Go test suite, so a test that truncates everything
# cannot wipe a running dev stack's data.
su - postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}_test'\"" | grep -q 1 || \
  su - postgres -c "createdb -O ${DB_USER} ${DB_NAME}_test"

# Report the port the cluster actually bound. Debian picks 5433 when another
# cluster already owns 5432, and a hard-coded 5432 in the printed DSN sends every
# caller to a database that is not there.
PORT=$(su - postgres -c "psql -tAc 'SHOW port'" | tr -d '[:space:]')

echo
echo "KILN_DATABASE_URL=postgres://${DB_USER}:${DB_PASS}@127.0.0.1:${PORT}/${DB_NAME}?sslmode=disable"
echo "KILN_TEST_DATABASE_URL=postgres://${DB_USER}:${DB_PASS}@127.0.0.1:${PORT}/${DB_NAME}_test?sslmode=disable"
