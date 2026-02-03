import os
import sys
import subprocess
import platform

def find_compiler():
    # Try finding clang in PATH
    try:
        subprocess.run(["clang", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "clang"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try common install locations for LLVM
    common_paths = [
        r"C:\Program Files\LLVM\bin\clang.exe",
        r"C:\Program Files (x86)\LLVM\bin\clang.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return f'"{path}"'
            
    # Fallback to gcc
    try:
        subprocess.run(["gcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "gcc"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    return None

def compile_extension(source_files, output_name):
    compiler = find_compiler()
    if not compiler:
        print("Error: No C compiler (clang or gcc) found.")
        print("Please install LLVM/Clang or MinGW-w64.")
        sys.exit(1)
        
    print(f"Using compiler: {compiler}")
    
    # Determine extension based on OS
    if platform.system() == "Windows":
        ext = ".dll"
        flags = ["-shared", "-O3", "-march=native"] 
    else:
        ext = ".so"
        flags = ["-shared", "-fPIC", "-O3", "-march=native"]
        
    output_file = output_name + ext
    
    cmd = [compiler] + source_files + ["-o", output_file] + flags
    
    # Quote paths properly
    quoted_sources = [f'"{s}"' for s in source_files]
    quoted_output = f'"{output_file}"'
    
    cmd_str = f'{compiler} {" ".join(quoted_sources)} -o {quoted_output} {" ".join(flags)}'
    print(f"Executing: {cmd_str}")
    
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        print(f"Successfully compiled {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed with error code {e.returncode}")
        print("\nDiagnostic Tips:")
        print("1. If you see 'file not found' (e.g., stdio.h), you are missing standard C headers.")
        print("   Solution: Install MinGW-w64 (via MSYS2) or Visual Studio Build Tools.")
        print("2. If 'command not found', ensure clang/gcc is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    c_dir = os.path.join(os.getcwd(), "src", "c")
    sources = [
        os.path.join(c_dir, "gensys.c"),
        os.path.join(c_dir, "kalman.c")
    ]
    
    output = os.path.join(c_dir, "dsge_c_lib")
    
    compile_extension(sources, output)
