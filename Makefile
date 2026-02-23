STEPS      ?= 30000
VOCAB_SIZE ?= 1000
LR         ?= 5e-4
EVAL_EVERY ?= 500
WEIGHTS    ?= weights
DATA       ?= data

export STEPS VOCAB_SIZE LR EVAL_EVERY

# ---------------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------------

.PHONY: all train router infer clean help

## Build release binaries
all:
	cargo build --release

## Train all three domains in parallel, then build router
train: train-parallel router

## Train all three domains in parallel
train-parallel:
	@echo "Training swe, work, creative in parallel..."
	@$(MAKE) -j3 train-swe train-work train-creative

## Build router from all trained domain data
router:
	cargo run --bin build_router --release -- \
		--data-dir $(DATA) \
		--out $(WEIGHTS)

## Interactive inference REPL
infer:
	cargo run --bin infer --release -- --weights $(WEIGHTS) --interactive

# ---------------------------------------------------------------------------
# Per-domain training targets
# ---------------------------------------------------------------------------

.PHONY: train-swe train-work train-creative

train-swe:
	scripts/train_swe.sh

train-work:
	scripts/train_work.sh

train-creative:
	scripts/train_creative.sh

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

## Delete all weights and logs
clean:
	rm -rf $(WEIGHTS)/*.bin $(WEIGHTS)/*.tok

## Show this help
help:
	@echo "Usage: make [target] [VAR=value ...]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Variables (with defaults):"
	@echo "  STEPS=$(STEPS)      training steps per domain"
	@echo "  VOCAB_SIZE=$(VOCAB_SIZE)     BPE vocabulary size"
	@echo "  LR=$(LR)          learning rate"
	@echo "  EVAL_EVERY=$(EVAL_EVERY)    evaluate every N steps"
	@echo "  WEIGHTS=$(WEIGHTS)    output directory for weights"
	@echo "  DATA=$(DATA)        input directory for training data"
	@echo ""
	@echo "Examples:"
	@echo "  make train                          # train all, default params"
	@echo "  make train STEPS=50000              # more steps"
	@echo "  make train-swe                      # single domain only"
	@echo "  make router                         # rebuild router only"
	@echo "  make infer                          # interactive inference"
	@echo "  make infer WEIGHTS=weights_v2       # use different weights dir"
