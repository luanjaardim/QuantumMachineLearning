{
  description = "Dev Shell for python using Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    py = pkgs.python3Packages;
  in
  {

    devShells.${system}.default = pkgs.mkShell {

      buildInputs = [
        py.python
        # Used to create a python enviroment, we can use pip normally with this
        py.venvShellHook

        # python packages that don't need to be controlled by pip
        py.numpy
        py.pandas
        py.matplotlib
        py.scikit-learn

        # Any other package that my be needed in the shell can be added here too
      ];

      venvDir = "./.env"; # Path to the python envirotment
      # To work with pytorch
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

      # As the shellHook is used to activate the Python Enviroment we enter our shell after it
      postShellHook = "
        SHELL=$(which nu)
        exec $SHELL
      ";

    };

  };
}
