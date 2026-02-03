# C Compiler Installation Instructions for FEDJuliaDSGE

To enable the high-performance C extensions for the DSGE model, you need to install a C compiler. We recommend **WinLibs (GCC)** as it is portable and easy to install without administrator privileges.

## Option A: WinLibs (Recommended - Portable)

1.  **Download**:
    *   Go to [https://winlibs.com/](https://winlibs.com/)
    *   Look for the latest **UCRT** release (e.g., `GCC 13.x.x + LLVM/Clang/LLD/LLDB + MinGW-w64 UCRT`).
    *   Download the **Zip** archive (e.g., `winlibs-x86_64-posix-seh-gcc-13.2.0-llvm-17.0.6-mingw-w64ucrt-11.0.1-r1.zip`).

2.  **Install**:
    *   Extract the contents of the zip file to a folder, e.g., `C:\winlibs`.
    *   You should see a `bin` folder inside (e.g., `C:\winlibs\mingw64\bin`).

3.  **Add to PATH**:
    *   Open **Start Menu**, search for "env", and select **Edit the system environment variables**.
    *   Click **Environment Variables**.
    *   Under **User variables** (top section), find `Path` and click **Edit**.
    *   Click **New** and paste the path to the `bin` folder (e.g., `C:\winlibs\mingw64\bin`).
    *   Click **OK** on all dialogs.

4.  **Verify**:
    *   Open a new terminal (PowerShell or Command Prompt).
    *   Run `gcc --version`.
    *   If installed correctly, it should print the version.

## Option B: Visual Studio Build Tools

If you already have Visual Studio installed or prefer it:
1.  Open **Visual Studio Installer**.
2.  Modify your installation.
3.  Select **Desktop development with C++**.
4.  Ensure **MSVC ... C++ x64/x86 build tools** is checked.
5.  Install.

## Next Steps

Once you have installed the compiler, please run the following command in your terminal to build the extensions:

```bash
python compile.py
```

If successful, the model will automatically use the C extensions.
