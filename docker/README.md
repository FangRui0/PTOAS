Build:

```bash
docker build . -t ptoas:py3.11
# default to py3.11 to be compatible with readily-availble CANN images at
# https://quay.io/repository/ascend/cann?tab=tags & https://github.com/Ascend/cann-container-image/tree/main/cann

# optional, to change python version
docker build . -t ptoas:py3.12 --build-arg PY_VER=cp312-cp312
```

Use:

```bash
docker run --rm -it ptoas:py3.11 /bin/bash
```
