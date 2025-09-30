{
  nixpkgs ? import <nixpkgs> {
    config.allowUnfree = true; #config.cudaSupport = true;
  },
  pkgs ? nixpkgs.pkgs
}:
let

  mkDevContainerCommands = dir: name: [
    (pkgs.writeShellScriptBin "build-${name}-python-container" ''
      podman build -t ${name}-python-dev-machine:latest .
    '')

    (pkgs.writeShellScriptBin "launch-${name}-python-container" ''
      podman run -td --volume=${dir}:/home/dev/src/ \
        --network=host \
        --user $(id -u):$(id -g) --userns keep-id:uid=$(id -u),gid=$(id -g)\
        --name=${name}-python-dev ${name}-python-dev-machine:latest
    '')

    (pkgs.writeShellScriptBin "stop-${name}-python-container" ''
      podman stop ${name}-python-dev || true
      podman rm ${name}-python-dev || true
    '')
  ];


  buildInputs = with pkgs; [
    (mkDevContainerCommands "/home/hadi/dev/aitinker" "aitinker")
  ];

in
pkgs.mkShell {
  name = "aitinker";

  inherit buildInputs;
}

