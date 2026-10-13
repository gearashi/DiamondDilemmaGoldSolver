"""Build the optional dependency-free C propagation library locally."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent

def library_name():
    return 'constraint_native.dll' if os.name=='nt' else 'libconstraint_native.dylib' if sys.platform=='darwin' else 'libconstraint_native.so'

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

def build(*,compiler=None,output_dir=None):
    compiler=find_compiler(compiler);output=Path(output_dir or ROOT/'.native').resolve()
    environment=os.environ.copy();environment['PATH']=str(Path(compiler).resolve().parent)+os.pathsep+environment.get('PATH','')
    output.mkdir(parents=True,exist_ok=True);source=ROOT/'constraint_native.c'
    with tempfile.TemporaryDirectory(prefix='build-',dir=output) as temp:
        temporary=Path(temp)/library_name()
        command=[compiler,'-O3','-std=c99','-shared',str(source),'-o',str(temporary)]
        if os.name!='nt':command.insert(1,'-fPIC')
        if sys.platform=='darwin':command[command.index('-shared')]='-dynamiclib'
        if os.name=='nt':command.insert(1,'-static-libgcc')
        result=subprocess.run(command,capture_output=True,text=True,timeout=60,env=environment)
        if result.returncode:raise RuntimeError('Native compilation failed (exit '+str(result.returncode)+'):\n'+result.stdout+result.stderr)
        target=output/library_name();os.replace(temporary,target)
    manifest={'abi':1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'library':target.name,
              'platform':sys.platform,'machine':platform.machine(),'compiler':compiler,
              'command':command,'compiler_version':subprocess.run([compiler,'--version'],capture_output=True,text=True,timeout=10,env=environment).stdout.splitlines()[0]}
    (output/'build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {'library':str(target),'manifest':manifest}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compiler');parser.add_argument('--output-dir')
    args=parser.parse_args(argv);print(json.dumps(build(compiler=args.compiler,output_dir=args.output_dir),indent=2))
if __name__=='__main__':main()
