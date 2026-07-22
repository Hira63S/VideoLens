build:
	docker compose build

run-api:
	docker compose up

run-pipeline:
	docker compose run --rm videolens python pipeline.py --config configs/example.yaml

run-query:
	docker compose run --rm videolens python query.py --config configs/example.yaml --query "cars on a road" --n_results 5