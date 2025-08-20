{
  description = "ink";

  nixConfig = {
    extra-substituters = ["https://mistzzt.cachix.org"];
    extra-trusted-public-keys = [
      "mistzzt.cachix.org-1:Ie2vJ/2OCl4D/ifadJLqqd6X3Uj7J2bDqNmw8n1hAJc="
    ];
    allowUnfree = true;
  };

  inputs = {
    progsyn.url = "github:mistzzt/program-synthesis-nur";
    nixpkgs.follows = "progsyn/nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    progsyn,
  }: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];

    forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    project-name = "rink-eval";
  in {
    packages = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      synPkgs = progsyn.packages.${system};

      python = pkgs.python312.override {
        packageOverrides = self: super:
          with pkgs.python312Packages; {
            tree-sitter-scala = buildPythonPackage rec {
              pname = "tree_sitter_scala";
              pyproject = true;
              version = "0.24.0";
              src = pkgs.fetchFromGitHub {
                owner = "tree-sitter";
                repo = "tree-sitter-scala";
                rev = "v${version}";
                hash = "sha256-ZE+zjpb52hvehJjNchJYK81XZbGAudeTRxlczuoix5g=";
              };
              build-system = [
                setuptools
              ];
              pythonImportsCheck = ["tree_sitter_scala"];
              doCheck = false;
            };

            tree-sitter = buildPythonPackage rec {
              pname = "tree-sitter";
              version = "0.24.0";
              pyproject = true;
              src = pkgs.fetchFromGitHub {
                owner = "tree-sitter";
                repo = "py-tree-sitter";
                rev = "24fcfe0ed8cdd8cc49756a52cfc135d42e083378";
                hash = "sha256-fJekcuh2WjP97v90qVhx86dQJk2Fp1wbilYoBQbn4/g=";
                fetchSubmodules = true;
              };
              build-system = [setuptools];
              pythonImportsCheck = ["tree_sitter"];
              doCheck = false;
            };
          };
      };
    in {
      ${project-name} = pkgs.rustPlatform.buildRustPackage {
        pname = project-name;
        version = "0.3.0";
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;
        buildInputs = with pkgs; [cvc5];
        preConfigure = ''
          echo "pub const CVC5_PATH: &str =\"${pkgs.cvc5}/bin/cvc5\";" > $PWD/src/local_config.rs
        '';
        doCheck = false;
      };
      transpile = with python.pkgs;
        buildPythonPackage rec {
          pname = "transpile";
          version = "0.3.0";
          pyproject = true;
          src = ./transpile;
          build-system = [hatchling];
          dependencies = [tree-sitter tree-sitter-scala];
        };
      default = self.packages.${system}.${project-name};
    });

    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      synPkgs = progsyn.packages.${system};
    in {
      default = pkgs.mkShell {
        shellHook = ''

          # parsynt needs extra racket packages
          cp -r ${synPkgs.parsynt}/src/synthools $PWD/synthools || true
          chmod -R 777 $PWD/synthools

          raco pkg install --auto rosette $PWD/synthools

          mkdir -p ~/.local/share/racket/8.14/pkgs/rosette/bin/
          ln -s ${pkgs.z3}/bin/z3 ~/.local/share/racket/8.14/pkgs/rosette/bin/z3 || true
        '';

        packages = with pkgs; [
          gh
          jq
          iconv
          cvc5
          cmake

          # parsynt dependencies
          synPkgs.parsynt
          pkgs.racket

          self.outputs.packages.${system}.${project-name}
          self.outputs.packages.${system}.transpile

          (pkgs.python312.withPackages (ps:
            with ps; [
              pandas
              tabulate
              matplotlib
              seaborn
              numpy
            ]))
        ];

        RUST_LOG = "none,rink=debug";
      };
    });
  };
}
