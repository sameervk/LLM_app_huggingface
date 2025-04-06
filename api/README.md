## Docker Image Build

* See the README of the root repository

## Docker Container

* There are different ways to start a docker container once the image is built.
  1. Interactive mode: `docker run -it --name [container-name] [image] [command]`
     * `command`: must be an command starting an interactive shell such as `/bin/bash` or `python`
     * The container will be available to access from the terminal itself where it was started.
  2. Detached mode: `docker run -d --name [container-name] [image] OPTIONAL[command]`
     * In this case, the container will continue to run if there is a command already specified in the dockerfile which starts a process, e.g. an app or a terminal.
     * The container can be accessed via vs code editor.
  3. Default mode: `docker run --network host --name [container-name] [image]`
     * To launch the app from the terminal, the `network` value must be set to `host`. Else, the app server cannot be accessed from the browser.
     * The default `network` value is `bridge` which enables connection between different containers.