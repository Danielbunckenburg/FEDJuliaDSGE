# Compile DSGE C Kernels on Windows
# Requires GCC (MinGW) and OpenBLAS/LAPACK

$srcDir = "C:\Users\Danie\OneDrive - University of Copenhagen\Programmering\FEDJuliaDSGE\src\c"
$binDir = "C:\Users\Danie\OneDrive - University of Copenhagen\Programmering\FEDJuliaDSGE\bin"

if (!(Test-Path $binDir)) {
    New-Item -ItemType Directory -Force -Path $binDir
}

# Note: This assumes gcc is in the PATH and OpenBLAS is available.
# Optimization flags:
# -O3: Maximum optimization
# -fopenmp: Enable parallel processing
# -march=native: Optimize for the current machine's CPU
gcc -shared -O3 -fopenmp -march=native -o "$binDir\dsge_kernels.dll" "$srcDir\gensys.c" "$srcDir\kalman.c" -lm -lblas -llapack

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful: $binDir\dsge_kernels.dll"
}
else {
    Write-Error "Compilation failed."
}
