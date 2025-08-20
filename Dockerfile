FROM nixpkgs/nix-flakes:nixos-24.11

WORKDIR /workspace
ENTRYPOINT ["nix", "develop"]
