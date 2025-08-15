.PHONY:
	docker-build-push-prod-remote \
	docker-build-dev-local \
	attach-gvirtus-dev \
	run-gvirtus-dev \
	run-gvirtus-dev-no-gpu \
	run-gvirtus-tests \

docker-build-push-prod-remote:
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--push \
		--no-cache \
		-f docker/prod/Dockerfile \
		-t gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-dev-local:
	docker buildx build \
		--platform linux/$$(uname -m) \
		--load \
		-f docker/dev/Dockerfile \
		-t gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04 \
		.

attach-gvirtus-dev:
	docker exec -it gvirtus bash

run-gvirtus-dev:
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
		-v ./docker/dev/build.sh:/build.sh \
		-v ./examples:/gvirtus/examples/ \
		--network host \
		--entrypoint /build.sh \
		--name gvirtus \
		--runtime=nvidia \
		--shm-size=8G \
		gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04

run-gvirtus-dev-no-gpu:
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
		-v ./docker/dev/build.sh:/build.sh \
		-v ./examples:/gvirtus/examples/ \
		--entrypoint /build.sh \
		--name gvirtus-no-gpu \
		--shm-size=8G \
		gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04

run-gvirtus-tests:
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
		-v ./docker/dev/build_and_test.sh:/build_and_test.sh \
		-v ./examples:/gvirtus/examples/ \
		--entrypoint /build_and_test.sh \
		--name gvirtus \
		--runtime=nvidia \
		--shm-size=8G \
		gvirtus-dev:cuda12.6.3-cudnn-ubuntu22.04