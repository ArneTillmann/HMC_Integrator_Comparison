{ pkgs ? import <nixpkgs> {} }:
with pkgs.python38Packages;
with pkgs;
mkShell {
  buildInputs = [ jupyter numpy jupytext ];
}
