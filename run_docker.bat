@echo off
Docker run -it --rm -p 8084:8084 -v ./mount:/usr/src/myapp/mount -v ./src:/usr/src/myapp/src rust_tch
