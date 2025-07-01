.PHONY: docker-build-push-dev docker-build-push-prod run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus

docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-push-prod:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/prod/Dockerfile \
		-t taslanidis/gvirtus:cuda12.6.3-cudnn-ubuntu22.04 \
		.

run-gvirtus-backend-dev:
	docker run \
		--rm \
		-it \
		-v ./cmake:/gvirtus/cmake/ \
		-v ./etc:/gvirtus/etc/ \
		-v ./include:/gvirtus/include/ \
		-v ./plugins:/gvirtus/plugins/ \
		-v ./src:/gvirtus/src/ \
		-v ./tools:/gvirtus/tools/ \
		-v ./tests:/gvirtus/tests/ \
		-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
		-v ./docker/dev/entrypoint.sh:/entrypoint.sh \
		-v ./examples:/gvirtus/examples/ \
		--entrypoint /entrypoint.sh \
		--name gvirtus \
		--gpus all \
		taslanidis/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04

run-gvirtus-tests:
	docker exec \
		-it gvirtus \
		bash -c \
		'export LD_LIBRARY_PATH=$$GVIRTUS_HOME/lib/frontend:$$LD_LIBRARY_PATH && \
			cd /gvirtus/build && \
			ctest --output-on-failure'

stop-gvirtus:
	docker stop gvirtus