# Reproduce everything in the repository.
#
#   make install     install the package with all extras
#   make test        run the test suite
#   make reproduce   regenerate every figure, CSV and paper number from the
#                    trained models in models/ (~10 min on one core)
#   make paper       build paper/main.pdf (needs a LaTeX distribution)
#   make train       retrain every RL agent from scratch (~2 h on one core;
#                    results will differ slightly from the committed models)
#
# Run `make -n <target>` to see the commands without running them.

PYTHON ?= python
S := scripts

.PHONY: install test reproduce figures lqr evaluate paper-assets paper train clean-paper

install:
	$(PYTHON) -m pip install -e ".[dev,rl]"

test:
	$(PYTHON) -m pytest -q

reproduce: test figures lqr evaluate paper-assets

# Phases 1-2: dynamics, Lagrange points, uncontrolled escape (seconds)
figures:
	$(PYTHON) $(S)/01_lagrange_points.py
	$(PYTHON) $(S)/02_jacobi_validation.py
	$(PYTHON) $(S)/03_l1_escape.py

# Phase 3: LQR baseline and its fuel/accuracy sweep (~1 min)
lqr:
	$(PYTHON) $(S)/04_lqr_station_keeping.py
	$(PYTHON) $(S)/05_lqr_pareto.py

# Phases 4-5: test-seed evaluation of the trained agents (~9 min)
evaluate:
	$(PYTHON) $(S)/07_compare_controllers.py --model dqn_l1
	$(PYTHON) $(S)/07_compare_controllers.py --model dqn_l1_run2
	$(PYTHON) $(S)/07_compare_controllers.py --model dqn_l1_run2 ppo_l1 ppo_l1_linear --out phase5
	$(PYTHON) $(S)/09_robustness.py --model dqn_l1_run2 ppo_l1
	$(PYTHON) $(S)/09_robustness.py --model ppo_l1 ppo_l1_s1 ppo_l1_s2 --out robustness_seeds
	$(PYTHON) $(S)/10_seed_study.py

# Phase 6: every number and table in the paper, from experiments/*.csv
paper-assets:
	$(PYTHON) $(S)/11_paper_assets.py

paper: paper-assets
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-paper:
	cd paper && latexmk -C

# Retraining overwrites models/ and the training logs. Each PPO run takes ~11 min
# with 8 environments; DQN runs ~22 min each.
train:
	$(PYTHON) $(S)/06_train_dqn.py --steps 1000000
	$(PYTHON) $(S)/06_train_dqn.py --steps 1000000 --lr 1e-4 --target-update 10000 \
		--w-pos 0.3 --name dqn_l1_run2
	for seed in 0 1 2; do \
		suffix=$$( [ $$seed -eq 0 ] && echo "" || echo "_s$$seed" ); \
		$(PYTHON) $(S)/08_train_ppo.py --steps 3000000 --seed $$seed --name ppo_l1$$suffix; \
		$(PYTHON) $(S)/08_train_ppo.py --steps 3000000 --seed $$seed --action-power 1 \
			--name ppo_l1_linear$$suffix; \
	done
