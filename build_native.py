"""Build the optional dependency-free C propagation library locally."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,sysconfig,tempfile
ROOT=Path(__file__).resolve().parent
WINDOWS_KERNELS = {
    'anneal_native': 'anneal-build.json',
    'constraint_native': 'build.json',
    'constraint_native_search': 'search-build.json',
}

def native_directory(manifest_name, root=ROOT):
    """Use shipped x64 libraries on Windows, retaining local builds elsewhere."""
    root = Path(root)
    packaged = root / 'native' / 'windows-x64'
    if (sysconfig.get_platform() == 'win-amd64' and manifest_name in WINDOWS_KERNELS.values()
            and packaged.is_dir()):
        return packaged
    return root / '.native'

def install_windows_bundle(root=ROOT):
    """Verify the shipped kernels, then activate the existing exact-engine loaders."""
    if sysconfig.get_platform() != 'win-amd64':
        raise RuntimeError('The packaged kernels require 64-bit x86 Windows Python.')
    root = Path(root)
    packaged = root / 'native' / 'windows-x64'
    pending = []
    for stem, manifest in WINDOWS_KERNELS.items():
        dll = packaged / (stem + '.dll')
        info = json.loads((packaged / manifest).read_text(encoding='utf-8'))
        source = root / (stem + '.c')
        if (info.get('abi') != 1 or info.get('platform') != 'win32'
                or info.get('machine') != 'AMD64' or info.get('library') != dll.name
                or info.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                or info.get('library_sha256') != hashlib.sha256(dll.read_bytes()).hexdigest()):
            raise RuntimeError(f'Windows kernel verification failed for {stem}. Re-extract the kit.')
        if stem != 'anneal_native':
            pending.extend([dll, packaged / manifest])
    # Existing exact-engine readers and saved source bindings use .native.
    # Preserve other files and avoid replacing identical DLLs in use.
    output = root / '.native'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        for source in pending:
            destination = output / source.name
            data = source.read_bytes()
            if destination.is_file() and destination.read_bytes() == data:
                continue
            staged = Path(tmp) / source.name
            staged.write_bytes(data)
            staged.replace(destination)

def build_windows_bundle(output, compiler=None):
    """Rebuild the three portable MinGW-w64 DLLs shipped with the Windows kit."""
    compiler = find_compiler(compiler or shutil.which('x86_64-w64-mingw32-gcc') or
                             shutil.which('gcc') or 'C:/msys64/mingw64/bin/gcc.exe')
    env = dict(os.environ, PATH=str(Path(compiler).resolve().parent) + os.pathsep + os.environ['PATH'])
    def identify(option):
        return subprocess.check_output([compiler, option], text=True, env=env).splitlines()[0]
    if identify('-dumpmachine') != 'x86_64-w64-mingw32':
        raise RuntimeError('The Windows bundle requires an x86_64-w64-mingw32 GCC toolchain.')
    version = identify('--version')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # GNU ld otherwise derives the preferred DLL base from the temporary output path.
    flags = ['-std=c99', '-O3', '-shared', '-static-libgcc', '-s',
             '-Wl,--no-insert-timestamp,--disable-auto-image-base']
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        for stem, manifest in WINDOWS_KERNELS.items():
            name = stem + '.dll'
            binary = Path(tmp) / name
            source = ROOT / (stem + '.c')
            subprocess.run([compiler, *flags, str(source), '-o', str(binary)], check=True, env=env)
            info = {'abi': 1, 'platform': 'win32', 'machine': 'AMD64', 'library': name,
                    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'library_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                    'compiler_version': version, 'flags': flags}
            (Path(tmp) / manifest).write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8', newline='\n')
        for path in Path(tmp).iterdir():
            path.replace(output / path.name)

def library_name(stem='constraint_native'):
    return stem+'.dll' if os.name=='nt' else 'lib'+stem+'.dylib' if sys.platform=='darwin' else 'lib'+stem+'.so'

def find_compiler(requested=None):
    if requested:
        value=shutil.which(requested) or (requested if Path(requested).is_file() else None)
        if value:return str(value)
        raise RuntimeError('Requested C compiler was not found.')
    for name in ('clang','gcc','cc'):
        found=shutil.which(name)
        if found:return found
    if os.name=='nt':
        for path in (Path('C:/msys64/mingw64/bin/gcc.exe'),Path('C:/msys64/ucrt64/bin/gcc.exe'),Path('C:/Program Files/LLVM/bin/clang.exe')):
            if path.is_file():return str(path)
    raise RuntimeError('Install GCC or Clang to build optional native propagation; Python fallback remains available.')

def _build_one(*,compiler=None,output_dir=None,stem='constraint_native',manifest_name='build.json'):
    compiler=find_compiler(compiler);output=Path(output_dir or ROOT/'.native').resolve()
    environment=os.environ.copy();environment['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+environment.get('PATH','')
    output.mkdir(parents=True,exist_ok=True);source=ROOT/(stem+'.c')
    with tempfile.TemporaryDirectory(prefix='build-',dir=output) as temp:
        temporary=Path(temp)/library_name(stem)
        command=[compiler,'-O3','-std=c99','-shared',str(source),'-o',str(temporary)]
        if os.name!='nt':command.insert(1,'-fPIC')
        if sys.platform=='darwin':command[command.index('-shared')]='-dynamiclib'
        if os.name=='nt':command.insert(1,'-static-libgcc')
        result=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
        if result.returncode:raise RuntimeError('Native compilation failed (exit '+str(result.returncode)+'):\n'+result.stdout+result.stderr)
        target=output/library_name(stem);os.replace(temporary,target)
    manifest={'abi':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'library':target.name,
              'platform':sys.platform,'machine':platform.machine(),'compiler':compiler,
              'command':command,'compiler_version':subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=10,env=environment).stdout.splitlines()[0]}
    (output/manifest_name).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {'library':str(target),'manifest':manifest}

def build(*,compiler=None,output_dir=None):
    # Preserve the propagation manifest/ABI and build search as a separate DLL.
    result=_build_one(compiler=compiler,output_dir=output_dir)
    result['search']=_build_one(compiler=compiler,output_dir=output_dir,
                              stem='constraint_native_search',manifest_name='search-build.json')
    result['physical_board']=_build_one(compiler=compiler,output_dir=output_dir,
                                       stem='r35_board_codec_decode',manifest_name='physical-board-build.json')
    result['board_delta']=_build_one(compiler=compiler,output_dir=output_dir,
                                    stem='dfs_checkpoint_decode',manifest_name='board-build.json')
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compiler')
    destination=parser.add_mutually_exclusive_group()
    destination.add_argument('--output-dir')
    destination.add_argument('--windows-bundle', metavar='DIRECTORY', help='Build redistributable Windows x64 libraries')
    destination.add_argument('--install-windows-bundle', action='store_true', help='Activate verified Windows x64 libraries')
    args=parser.parse_args(argv)
    if args.install_windows_bundle:
        install_windows_bundle()
        print('Verified Windows x64 annealing, propagation and search kernels.')
    elif args.windows_bundle:
        build_windows_bundle(args.windows_bundle, args.compiler)
        print('Built Windows x64 libraries in', args.windows_bundle)
    else:
        print(json.dumps(build(compiler=args.compiler,output_dir=args.output_dir),indent=2))
if __name__=='__main__':main()
