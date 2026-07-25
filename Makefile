.DEFAULT_GOAL := help

up:  ## Sobe o container com docker compose
	docker compose up -d

reload:  ## Reconstrói a imagem e recria o container
	docker compose up -d --build

down:  ## Derruba o container
	docker compose down

logs:  ## Mostra os logs em tempo real
	docker compose logs -f

help:  ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'
