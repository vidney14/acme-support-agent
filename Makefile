.PHONY: install api web offline test smoke calibrate evals

PY := backend/.venv/bin/python

install:            ## create backend venv + install frontend deps
	cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
	cd frontend && npm ci

api:                ## backend with your Gemini key (reads .env)
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

offline:            ## backend with NO key: rule-based agent + simulated voice
	cd backend && LLM_PROVIDER=fake .venv/bin/uvicorn app.main:app --reload --port 8000

web:                ## frontend on http://localhost:3000
	cd frontend && npm run dev

test:               ## all automated checks (offline)
	cd backend && .venv/bin/pytest -q
	cd backend && LLM_PROVIDER=fake .venv/bin/python -m scripts.run_scenarios
	cd frontend && npx tsc --noEmit && npm run lint

smoke:              ## verify every Gemini integration with your key (~1 min)
	cd backend && .venv/bin/python -m scripts.smoke_test

calibrate:          ## measure real similarity scores, suggest guardrail thresholds
	cd backend && .venv/bin/python -m scripts.calibrate_thresholds

evals:              ## end-to-end scenarios against the real model
	cd backend && .venv/bin/python -m scripts.run_scenarios
