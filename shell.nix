{ pkgs ? import <nixpkgs> {} }:

let
  # Собираем PyTorch только для CPU, без CUDA
  torch-cpu = pkgs.python3Packages.torch.override { cudaSupport = false; };
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      jupyter
      ipykernel
      numpy
      pandas
      matplotlib
    ]))
    torch-cpu
    pkgs.stdenv.cc.cc.lib   # libstdc++.so.6
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
