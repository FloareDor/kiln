# Kiln
#
# Everything GPU-related runs inside WSL2 on the development box, because vLLM has no
# Windows build. `scripts/go.sh` wraps the Go toolchain so its caches live on the WSL
# native filesystem -- building with GOCACHE on /mnt/c is roughly an order of
# magnitude slower because of 9p round-trips per object file.

SHELL := /bin/bash
GO    := bash scripts/go.sh
BIN   := bin

VLLM_VERSION ?= 0.32.0
KILN_TEST_DATABASE_URL ?= postgres://kiln:kiln@127.0.0.1:5433/kiln_test?sslmode=disable
KILN_DATABASE_URL      ?= postgres://kiln:kiln@127.0.0.1:5433/kiln?sslmode=disable
export KILN_TEST_DATABASE_URL KILN_DATABASE_URL

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------- build

.PHONY: build
build: ## Build all three binaries into bin/
	@mkdir -p $(BIN)
	$(GO) build -trimpath -o $(BIN)/gateway      ./cmd/gateway
	$(GO) build -trimpath -o $(BIN)/controlplane ./cmd/controlplane
	$(GO) build -trimpath -o $(BIN)/nodeagent    ./cmd/nodeagent
	@echo "built: $(BIN)/{gateway,controlplane,nodeagent}"

.PHONY: vet
vet: ## go vet
	$(GO) vet ./...

.PHONY: tidy
tidy: ## go mod tidy
	$(GO) mod tidy

# ------------------------------------------------------------------------ test

.PHONY: test
test: ## Unit tests only -- no GPU, no database, no network
	KILN_TEST_DATABASE_URL= $(GO) test ./...

.PHONY: test-all
test-all: db ## Every test, including the Postgres integration suite
	$(GO) test ./... -count=1

.PHONY: cover
cover: ## Test coverage summary
	KILN_TEST_DATABASE_URL= $(GO) test ./... -coverprofile=/tmp/kiln.cover
	$(GO) tool cover -func=/tmp/kiln.cover | tail -20

# -------------------------------------------------------------- local services

.PHONY: setup
setup: ## One-time: install vLLM, Go and Postgres inside WSL2
	bash scripts/setup_wsl_go.sh
	VLLM_VERSION=$(VLLM_VERSION) bash scripts/setup_wsl_vllm.sh
	bash scripts/setup_wsl_postgres.sh

.PHONY: db
db: ## Start Postgres and apply migrations
	@bash scripts/setup_wsl_postgres.sh >/dev/null
	@$(MAKE) --no-print-directory build
	@$(BIN)/controlplane -database-url "$(KILN_DATABASE_URL)" -migrate-only

# ------------------------------------------------------------------ milestones

.PHONY: m0
m0: ## M0 gate: prove vLLM sleep mode frees and restores GPU memory
	bash scripts/run_m0.sh

.PHONY: m1
m1: build ## M1: OpenAI SDK streams end to end
	bash scripts/run_e2e.sh m1

.PHONY: m2
m2: build db ## M2: one-call deploy, quotas, crash-safe jobs
	bash scripts/run_m2.sh

# ------------------------------------------------------------------ kubernetes

.PHONY: images
images: ## Build container images
	docker build -f deploy/docker/Dockerfile.nodeagent \
	  --build-arg VLLM_VERSION=$(VLLM_VERSION) -t kiln/nodeagent:dev .
	docker build -f deploy/docker/Dockerfile.services \
	  --build-arg TARGET=gateway -t kiln/gateway:dev .
	docker build -f deploy/docker/Dockerfile.services \
	  --build-arg TARGET=controlplane -t kiln/controlplane:dev .

.PHONY: cluster
cluster: ## Start a local GPU-enabled minikube cluster
	minikube start --driver docker --container-runtime docker --gpus all \
	  --cpus 8 --memory 16384
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
	kubectl label node --all nvidia.com/gpu.present=true --overwrite

.PHONY: up
up: images ## Deploy the whole stack to the current cluster
	@# Load images into minikube's daemon rather than pushing to a registry; without
	@# this the pods sit in ImagePullBackOff looking for kiln/gateway on Docker Hub.
	-minikube image load kiln/nodeagent:dev kiln/gateway:dev kiln/controlplane:dev
	kubectl apply -f deploy/k8s/kiln.yaml
	kubectl apply -f deploy/k8s/monitoring.yaml
	kubectl -n kiln rollout status deploy/kiln-gateway --timeout=300s
	kubectl -n kiln rollout status deploy/kiln-controlplane --timeout=300s
	kubectl -n kiln rollout status ds/kiln-nodeagent --timeout=600s
	@echo
	@echo "gateway:  http://$$(minikube ip):30080"
	@echo "grafana:  http://$$(minikube ip):30300  (admin/admin)"

.PHONY: down
down: ## Remove the stack
	-kubectl delete -f deploy/k8s/monitoring.yaml --ignore-not-found
	-kubectl delete -f deploy/k8s/kiln.yaml --ignore-not-found

.PHONY: logs
logs: ## Tail node agent logs
	kubectl -n kiln logs -f ds/kiln-nodeagent

# ----------------------------------------------------------------- benchmarks

.PHONY: traces
traces: ## Generate the fixed benchmark traces
	python bench/tracegen.py --out bench/traces

.PHONY: bench
bench: ## Run an experiment: make bench EXP=3
	python bench/run_experiment.py --experiment $(EXP)

.PHONY: clean
clean: ## Remove build output
	rm -rf $(BIN) /tmp/kiln.cover
