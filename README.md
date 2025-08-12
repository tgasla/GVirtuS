# GVirtuS

The GPU Virtualization Service (GVirtuS) presented in this work tries to fill the gap between in-house hosted computing clusters, equipped with GPGPUs devices, and pay-for-use high performance virtual clusters deployed via public or private computing clouds. gVirtuS allows an instanced virtual machine to access GPGPUs in a transparent and hypervisor independent way, with an overhead slightly greater than a real machine/GPGPU setup. The performance of the components of gVirtuS is assessed through a suite of tests in different deployment scenarios, such as providing GPGPU power to cloud computing based HPC clusters and sharing remotely hosted GPGPUs among HPC nodes.

**Read the official GVirtuS paper [here](https://link.springer.com/chapter/10.1007/978-3-642-15277-1_37).**

# 📄 Published Papers

You can view the full list of all GVirtuS published papers in [CITATIONS](CITATIONS.md).

# Method 1: 🧰 Install from Source Directly on a Host Machine

## Prerequisites

**Tested OS**: Ubuntu 22.04 LTS
    - In principle, every OS that can support CUDA, can also run GVirtuS

Before proceeding, ensure the following dependencies are installed on your system:

```bash
sudo apt update && sudo apt install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa6-dev liblog4cplus-dev librdmacm-dev libibverbs-dev libgtest-dev liblz4-dev cmake git
```

* [CUDA Toolkit & CUDA Driver](https://developer.nvidia.com/cuda-downloads): _Tested with CUDA Toolkit **v12.6.3** (latest verified working version)_

* [cuDNN](https://developer.nvidia.com/cudnn-downloads): _Tested with cuDNN **v9.5.1** (latest verified working version)_

## Clone the Repository and `cd` into the GVirtuS folder

```bash
git clone https://github.com/tgasla/gvirtus.git && cd gvirtus
```

## Create a build folder and `cd` into it

```bash
mkdir build && cd build
```

## Setup the GVirtuS Environment Variables

### GVIRTUS_HOME

GVirtuS will be installed in the `GVIRTUS_HOME` environment variable path. A good place to install GVirtuS is either your home directory or inside `/usr/local/gvirtus`

```bash
export GVIRTUS_HOME=/usr/local/gvirtus
```

### GVIRTUS_LOGLEVEL

GVirtuS uses log4cplus as its logging system. You can set the loglevel to be any of the following:
- 0: for TRACE
- 10000: for DEBUG
- 20000: for INFO
- 30000: for WARN
- 40000: for ERROR
- 50000: for FATAL

For example:

```bash
export GVIRTUS_LOGLEVEL=20000
```

## Install

```bash
cmake .. && make && make install
```

## Edit the properties.json (if needed)

The GVirtuS backend will listen to the endpoint (address:port) specified in `${GVIRTUS_HOME}/etc/properties.json` configuration file. Similarly, the GVirtuS frontend will try to conenct to the endpoint specified in the same file.

You can directly edit this file with the `server_address` and `port` you want using the following commands:

```bash
sed -i 's/"server_address": "0.0.0.0"/"server_address": "DESIRED_SERVER_ADDRESS"/' ${GVIRTUS_HOME}/etc/properties.json
```

and

```bash
sed -i 's/"port": "9999"/"port": "DESIRED_PORT_NUMBER"/' ${GVIRTUS_HOME}/etc/properties.json
```

Similarly, you can modify other configuration options in `properties.json`, such as the communication protocol used between the GVirtuS backend and frontend, or specify which plugins to be loaded.

If you prefer editing the file manually, you can use text editors like `vim` or `nano`. Note that these editors are not pre-installed in the Docker containers, so you need to install them first if required.

## Configure and Run the GVirtuS backend

First, set up the `LD_LIBRARY_PATH` environment variable:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}
```

Then, to run the GVirtuS backend (on a machine that has access to a physical GPU):

```bash
${GVIRTUS_HOME}/bin/gvirtus_backend ${GVIRTUS_HOME}/etc/properties.json
```

The above command will use `${GVIRTUS_HOME}/etc/properties.json` configuration file.

## Configure the GVirtuS frontend

First, set up the `LD_LIBRARY_PATH` environment variable:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
```

Then, you can compile and run any application you want that utilizes cuda and GVirtuS will be used.

If the application is written in python, the above is not enough, you will also need to prefix your python application with the `LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':')`.

For example, if you want to run a python script named `app.py`, instead of doing `python3 app.py`, you should do:

```bash
LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') python3 app.py
```

> [!WARNING]
> If you use a python manager like pyenv, the above will not work because `python3` is not the python binary. It is ASCII text executable managed by pyenv. Instead, you should either turn off pyenv or manually use the real python binary using:

```bash
LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') /usr/bin/python3 app.py
```

# Method 2: 🐳 Use the Official GVirtuS Docker Image (Recommended)

GVirtuS provides official Docker images on [Docker Hub](https://hub.docker.com/u/gvirtus) for both `linux/amd64` and `linux/arm64` platforms. Using these images is the fastest and most reliable way to accelerate your workflow, especially if you have fixed applications that you set up once and use repeatedly.

## Prerequisites

- Install Docker in both your GVirtuS frontend and backend machines.
- Install NVIDIA Container Toolkit only on your GVirtuS backend machines.

To do so, follow the instructions [here](#prerequisites-2).

## Download GVirtuS Docker image

```bash
docker pull gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

In the Docker image, the environment variable `GVIRTUS_HOME` is set to `/usr/local/gvirtus`, indicating that GVirtuS is installed in that directory.

## Change the GVirtuS Backend Server Address and Port Number (if needed)

By default, the Docker image configures the server address as `0.0.0.0` (listening on all interfaces) and the port as `9999`. If these settings are suitable for your environment, you can start the GVirtuS backend immediately with:

```bash
docker run -it --rm --name gvirtus-backend --network host --runtime nvidia gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

If you want your GVirtuS backend server to listen on a different address or port, follow these steps:

- Create a `properties.json` file anywhere in your host machine similar to the one one in `etc/properties.json`.
- Change the `server_address` and `port` fields as needed and save the file.
- Then, assuming you run the `docker run` command in the same directory as your `properties.json` file, you can overrive the original `properrties.json` configuration file inside the container by doing:

```bash
docker run -it --rm --name gvirtus-backend --network --runtime nvidia host -v ./properties.json:/usr/local/gvirtus/etc/properties.json gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

The original image `GVIRTUS_LOGLEVEL` has been set to 20000 (INFO). If you wish to change that, you can do so by overriding the environment variable. For example, if you want to start the GVirtuS backend server using `DEBUG` loglevel you can do:

```bash
docker run -it --rm --name gvirtus-backend --network host --runtime nvidia -v ./properties.json:/usr/local/gvirtus/etc/properties.json -e GVIRTUS_LOGLEVEL=0 gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04
```

More information on the available log levels can be found [here](#gvirtus_loglevel).

## Run your GVirtuS Frontend Application in a Docker Container

First, create a Dockerfile based on GVirtuS image. Your Dockerfile, should follow the template below.

```Dockerfile
FROM gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04

RUN sed -i 's/"server_address": "0.0.0.0"/"server_address": "BACKEND_SERVER_ADDRESS"/' ${GVIRTUS_HOME}/etc/properties.json
RUN sed -i 's/"port": "9999"/"port": "BACKEND_PORT_NUMBER"/' ${GVIRTUS_HOME}/etc/properties.json

RUN apt update && apt install -y <APPLICATION_DEPENDENCIES_IF_ANY>
...
<DO_ANY_OTHER_SETUP_STEPS_NEEDED>
...
RUN <COMPILE_YOUR_APPLICATION_IF_NEEDED>

ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

ENTRYPOINT ["<YOUR_APPLICATION>"]
CMD ["<default_application_argument_1>, <default_application_argument_2>"]
```

- If your application does not require arguments, you can omit the `CMD` directive from your Dockerfile.
- For Python applications, set the `ENTRYPOINT` to include the `LD_PRELOAD` environment variable before the Python executable to ensure GVirtuS frontend libraries are loaded. Refer to the [Configure the GVirtuS frontend](#configure-the-gvirtus-frontend) section for details.

Then, build your docker image using:

```bash
docker build -t gvirtus-app .
```

Run your application using:

```bash
docker run -it --rm --name gvirtus-app gvirtus-app
```

For example, a Dockerfile for a minimal pytorch application is given below:

```Dockerfile
# Dockerfile

FROM gvirtus/gvirtus:cuda12.6.3-cudnn-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu126 numpy

COPY ./test_pytorch.py /test_pytorch.py
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}

RUN sed -i 's/"server_address": "0.0.0.0"/"server_address": "<YOUR_BACKEND_SERVER_ADDRESS>"/' ${GVIRTUS_HOME}/etc/properties.json
RUN sed -i 's/"port": "9999"/"port": "<YOUR_BACKEND_PORT>"/' ${GVIRTUS_HOME}/etc/properties.json

ENTRYPOINT ["/bin/bash", "-c", "LD_PRELOAD=$(echo ${GVIRTUS_HOME}/lib/frontend/*.so | tr ' ' ':') /usr/bin/python3 /test_pytorch.py"]
```

Note that some libraries may need to be built with specific flags, such as setting the CUDA library path to the GVirtuS libraries path and disabling the use of the static cudart library.

For example, when building libraries like OpenCV or OpenPose (and possibly others), you should add the following flag to your CMake command:

```bash
-D CUDA_USE_STATIC_CUDA_RUNTIME=OFF
```

Additionally, as shown in the Dockerfile example above, some libraries require you to explicitly enable CUDA support during installation. By default, CUDA support may be disabled to reduce installation time and resource usage, so be sure to specify the appropriate options if you need GPU acceleration.

The `test_pytorch.py` is also given below:

```python
# test_pytorch.py

import torch

device = torch.device("cuda")

model = torch.nn.Linear(10, 1).to(device)

x = torch.randn(5, 10).to(device)

output = model(x)

print(output)
```

You can also create a simple bash script to serve as your application's entrypoint and specify it with the `--entrypoint` flag in the `docker run` command.

If you are actively developing your application and want to avoid rebuilding the Docker image every time you make code changes, use the `-v` mount volume option with `docker run`. This allows you to mount your source code directory into the container, making development more efficient.

If the GVirtuS backend server address or port number changes frequently, hardcoding these values in your Docker frontend image can be inconvenient. Instead, create a `properties.json` configuration file on your host machine and mount it into the container using the `-v` flag, as described for the backend [above](#change-the-gvirtus-backend-server-address-and-port-number-if-needed). This approach keeps your configuration flexible and easy to update.

# Method 3: 🧰 🐳 Install from Source in a Docker Container (Debugging/Interactive Mode)

This method is ideal for development and debugging GVirtuS. It lets you build and run GVirtuS from source interactively inside a Docker container, making it easy to test code changes, troubleshoot issues, and experiment with the GVirtuS environment in real time.

## Prerequisites
* [Docker](https://docs.docker.com/engine/install/): _Ensure Docker Engine is properly installed and running_
    - On Ubuntu:
        ```bash
        sudo apt install -y docker.io
        sudo usermod -aG docker $USER
        newgrp docker
        ```

* [Docker Buildx Plugin](https://github.com/docker/buildx#installing): _Used in the Makefile targets that build the GVirtuS docker images_
    - On Ubuntu: 
        ```bash
        sudo apt install -y docker-buildx-plugin
        ```

* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): _Install only on GPU nodes that will run the GVirtuS backend_
    - On Ubuntu:
        ```bash
        sudo apt install -y nvidia-container-toolkit
        ```

## `git clone` the **GVirtuS** main repository: 

```bash
git clone https://github.com/tgasla/gvirtus.git
```

## `cd` into the repo directory:

```bash
cd gvirtus
```

## Build the GVirtuS docker image locally

```bash
make docker-build-dev-local
```

## Run the GVirtuS image with GPU access (GVirtuS backend)

```bash
make run-gvirtus-dev
```

This command launches a Docker container and mounts your local repository into it. Any changes you make to your files on your host machine will be instantly available inside the container, enabling rapid development and testing without needing to rebuild the image.

Reminder: After creating the GVirtuS backend container, you can immediately launch the GVirtuS backend as described above, or make any configuration or/and environment variable changes before starting it. The `LD_LIBRARY_PATH` is already set up correctly for the backend, so you do not need to change it.

## Run the GVirtuS image without GPU access (GVirtuS frontend)

```bash
make run-gvirtus-dev-no-gpu
```

After launching a GVirtuS frontend, you must set the `LD_LIBRARY_PATH` to include the frontend libraries:

```bash
export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib/frontend:${LD_LIBRARY_PATH}
```

This is necessary because the Docker image (see [Dockerfile](/docker/dev/Dockerfile)) sets `LD_LIBRARY_PATH` to include only `${GVIRTUS_HOME}/lib`, which is sufficient only for the backend. The frontend also requires `${GVIRTUS_HOME}/lib/frontend` in the path to function correctly.

# Method 4: 🧪 Experimental/Preliminary Debugging

For even faster debugging, you can run both the GVirtuS backend and frontend inside the same Docker container. This approach simplifies setup since you only need one container instead of two.

**Workflow:**

1. **Start the container (backend session):**
    Open a terminal and run:
    ```bash
    make run-gvirtus-dev
    ```
    Use this shell for the GVirtuS backend. Configure (if needed) and launch as described above.

2. **Attach a second shell (frontend session):**
    In another terminal, attach to the same running container:
    ```bash
    make attach-gvirtus-dev
    ```
    Use this shell for the GVirtuS frontend. Configure (if needed), download and install dependencies (if any) and run your application.

> [!IMPORTANT]
> The backend and frontend require different `LD_LIBRARY_PATH` settings:
> - **Backend:**  
>   ```bash
>   export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib
>   ```
> - **Frontend:**  
>   ```bash
>   export LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${GVIRTUS_HOME}/lib/frontend
>   ```
> By default, the developer Docker image sets `LD_LIBRARY_PATH` for the backend only (see [Dockerfile](/docker/dev/Dockerfile)). In your frontend session, update it as [shown above](#run-the-gvirtus-image-without-gpu-access-gvirtus-frontend).

**Testing and Examples:**
- To run unit tests in the frontend session:
  ```bash
  cd /gvirtus/build
  ctest --output-on-failure
  ```
- To try GVirtuS [examples](/examples/):
  1. Ensure the backend is running in one shell.
  2. In the frontend shell, `cd` to an example directory.
  3. Install dependencies (usually with `./setup.sh`).
  4. Run the example (usually with `./run.sh`).

> [!CAUTION]
> Running both backend and frontend in the same container gives the frontend direct GPU access. This can lead to misleading results (an example may work in this setup but fail when the frontend does **not** have GPU access). Use this method for quick initial testing only.

> To verify your application works without GPU access, always test using a frontend container started with:
> ```bash
> make run-gvirtus-dev-no-gpu
> ```

> [!NOTE]
> Not all [examples](/examples/) have been tested with a frontend that does not have GPU access. Some examples may not function correctly in this scenario.

These sections contain information that also apply to the Docker installation:
- [Setup GVirtuS Environment Variables](#setup-the-gvirtus-environment-variables)
- [Edit the properties.json configuration file](#edit-the-propertiesjson-if-needed)
- [Configure and run the GVirtuS backend](#configure-and-run-the-gvirtus-backend)
- [Configure the GVirtuS frontend](#configure-the-gvirtus-frontend)

# 📊 GVirtuS Implementation Status

🗂️ Looking for function coverage? Check out the [**STATUS**](./STATUS.md) file for a detailed breakdown of which CUDA functions are:
- 🛠️ Implemented
- 🧪 Tested
- ⚙️ Working

This file tracks progress across major CUDA libraries (e.g., cuBLAS, cuDNN, cuRAND) and helps monitor GVirtuS coverage.

# 🔬 Testing GVirtuS

To test GVirtuS, follow the steps below. This setup runs the GVirtuS backend inside a Docker container with all dependencies pre-installed, and mounts your local source and test files into the container for easy development and debugging.

## Running GVirtuS unit tests

Use the script below to run the GVirtuS tests. It builds GVirtuS from source inside a Docker container and launches the backend process. This script creates a new process inside the same container that acts as the frontend and runs all test files located in the tests/ directory:

```bash
make run-gvirtus-tests
```

## Adding Tests

To add new tests, simply place your test code in any existing .cu file inside the tests directory. You can also create new .cu files if you wish; just make sure to include them as source files in [tests/CMakeLists.txt](tests/CMakeLists.txt#L24).

After making local changes to the GVirtuS source or tests, you can re-run the `make run-gvirtus-tests` command.

# ⚠️ Disclaimers

GVirtuS is currently not production-ready. It is **not thread-safe** and has known **memory leaks**. Use it with caution in experimental or non-critical environments.