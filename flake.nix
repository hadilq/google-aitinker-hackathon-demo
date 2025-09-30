{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = args@{ self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system; config.allowUnfree = true; config.cudaSupport = true;
        };
      in
      {
        devShells.default = import ./shell.nix { inherit nixpkgs pkgs; };
      }
    );
}

