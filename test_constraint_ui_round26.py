"""Executed dashboard controls and server parity for complete workstreams."""
import itertools,json,shutil,subprocess,unittest
from test_constraint_ui_round25 import HARNESS,Inputs,ROOT
from search_options import ROUND26_OPTIONS,BOOLEAN_OPTIONS,normalize_options,option_engines
from runtime_capabilities import validate_runtime_selection
from test_runtime_capabilities import capabilities

class Round26Controls(unittest.TestCase):
    def node(self,body):
        if not shutil.which('node'):self.skipTest('Node is needed for actual JavaScript controls.')
        result=subprocess.run([shutil.which('node'),'-e',HARNESS+body,str(ROOT/'Dashboard.html')],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    def test_registered_controls_defaults_dependencies_and_numeric_bound(self):
        parser=Inputs();parser.feed((ROOT/'Dashboard.html').read_text(encoding='utf-8'))
        self.assertEqual({name[4:] for name in parser.inputs if name.startswith('r26_')},set(ROUND26_OPTIONS))
        for key in ROUND26_OPTIONS:
            self.assertEqual(len(parser.inputs['r26_'+key]),1)
            self.assertNotIn('checked',parser.inputs['r26_'+key][0]);self.assertFalse(normalize_options()[key])
        self.assertEqual(parser.inputs['satServiceInstancesInput'][0]['max'],'8')
        cases=[{key:True} for key in ROUND26_OPTIONS]
        cases += [dict.fromkeys(keys,True) for keys in itertools.combinations(('durable_reasoning','mixed_proof_dag','proof_transport','proof_indexing','coordinated_sat','persistent_theory'),2)]
        self.node('const cases='+json.dumps([(case,{key:value for key,value in normalize_options({**dict.fromkeys(BOOLEAN_OPTIONS,False),**case}).items() if value is True}) for case in cases])+";for(const[input,expected]of cases){const actual=run('normalizeSearchOptions('+JSON.stringify(input)+')');assert.equal(JSON.stringify(Object.keys(actual).sort()),JSON.stringify(Object.keys(expected).sort()));for(const[key,value]of Object.entries(actual))assert.equal(value,expected[key],key)}")

    def test_coordinated_engine_controls_and_cuda_only_validation(self):
        caps=capabilities('pysat','cupy','wgpu','ortools')
        self.node('run('+json.dumps('state.capabilities='+json.dumps(caps)+';renderState();')+');'+r"""
select('dfs');element('backendInput').value='cuda';
element('r26_coordinated_sat').checked=true;element('r26_coordinated_gpu').checked=true;run('renderState()');
assert.equal(element('workersField').classList.contains('hidden'),false);
assert.equal(element('r26_sat_guard_reuse').disabled,false);
assert.equal(element('r26_gpu_support_reuse').disabled,false);
assert.equal(element('replicaInput').max,'512');assert.equal(element('replicaInput').value,'128');
element('r26_sat_guard_reuse').checked=true;element('r26_gpu_support_reuse').checked=true;
element('workersInput').value='2';element('satServiceInstancesInput').value='2';element('seconds').value='0';element('seedInput').value='17';
run("command('start')");const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.search_options.symbolic_frontier,true);assert.equal(sent.search_options.sat_guard_reuse,true);assert.equal(sent.search_options.gpu_support_reuse,true);assert.equal(sent.workers,2);assert.equal(sent.seconds,0);
idle();element('backendInput').value='webgpu';run('renderState()');assert.equal(element('startBtn').disabled,true);
""")
        with self.assertRaisesRegex(ValueError,'CUDA'):validate_runtime_selection('dfs','webgpu',normalize_options({'coordinated_gpu':True}),capabilities=caps)
        with self.assertRaisesRegex(ValueError,'proof logging'):validate_runtime_selection('dfs','cuda',normalize_options({'coordinated_sat':True,'persistent_theory':True,'native_sat_persistence':True}),capabilities=caps)

    def test_structural_gpu_proofs_do_not_require_optional_sat_oracle(self):
        caps=capabilities('cupy','wgpu')
        expression='round25OptionReason("gpu_shared_clauses","gpu-domains",{gpu_proof_transport:true},'+json.dumps(caps)+')'
        self.node('assert.equal(run('+json.dumps(expression)+'),"");')
        expression='round25OptionReason("gpu_shared_clauses","gpu-domains",{},'+json.dumps(caps)+')'
        self.node('assert.notEqual(run('+json.dumps(expression)+'),"");')

    def test_live_coordinator_work_units_are_separate(self):
        self.node(r"""
run(`state={process_running:true,status:{state:'running',method:'constraint',algorithm:'parallel-dfs',replicas:128,workers:2,search_options:{symbolic_frontier:true,coordinated_gpu:true},stats:{coordinator_worker_processes:3,coordinator_live_leases:2,coordinator_jobs_closed:7,coordinator_cpu_decisions:19,coordinator_sat_decisions:23,coordinator_gpu_decisions:29,coordinator_slices:5}},history:[]};renderState();`);
assert.equal(element('replicasLabel').textContent,'Worker processes');assert.equal(element('replicas').textContent,'3');assert.ok(element('replicaNote').textContent.includes('2 leased jobs'));assert.ok(element('replicaNote').textContent.includes('128 CUDA lanes'));
""")

if __name__=='__main__':unittest.main()
