"""Actual dashboard JavaScript controls, dependencies, scope and availability."""
import json
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path
from search_options import ROUND25_OPTIONS, normalize_options

ROOT=Path(__file__).resolve().parent
KEYS=('sat_core_lifting','shared_sat_assistant','resumable_proof_checking',
      'native_explanations','symbolic_coverage','mixed_proof_dag','higher_order_relations','incremental_affine',
      'native_sat_persistence','conflict_pb','integer_lattice','hall_rematch','reversible_gold_graph',
      'incremental_regions','durable_regions','compact_factors','gold_elimination',
      'separator_search','selective_refinement','joint_symbolic_regions','algebraic_connectivity','residual_isomorphism','gold_pricing',
      'multichain_routing','region_certificates')
HARNESS=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'4096',style:{},dataset:{},classList:{values:new Set(),add(name){this.values.add(name)},remove(name){this.values.delete(name)},contains(name){return this.values.has(name)},toggle(name,force){const add=force===undefined?!this.values.has(name):force;if(add)this.values.add(name);else this.values.delete(name)}},setAttribute(){},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},clientWidth:500,textContent:'',innerHTML:'',checked:true});return elements.get(id)}
const storage={values:{},getItem(key){return this.values[key]??null},setItem(key,value){this.values[key]=value}};
const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{addEventListener(){},localStorage:storage},console,Date,Number,String,Math,Set,JSON,setTimeout(){return 1},clearTimeout(){},setInterval(){},fetchCalls:[]});
context.fetch=(url,options)=>{context.fetchCalls.push([url,options]);return new Promise(()=>{})};
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0].replace('poll();setInterval(poll,1000);','');
vm.runInContext(source,context);
function run(source){return vm.runInContext(source,context)}
function select(algorithm){element('algorithmInput').value=algorithm;element('algorithmInput').listeners.change()}
function edit(key,value){element('r25_'+key).checked=value;element('r25_'+key).listeners.change()}
function idle(){run("commandBusy=false;state={process_running:false,control_token:'test',status:{state:'stopped',method:'constraint',algorithm:'dfs'},history:[]};renderState();");}
idle();
"""


class Inputs(HTMLParser):
    def __init__(self):super().__init__();self.inputs={}
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='input' and 'id' in attrs:self.inputs.setdefault(attrs['id'],[]).append(attrs)


class Round25Controls(unittest.TestCase):
    def test_controls_match_only_implemented_options_and_default_off(self):
        source=(ROOT/'Dashboard.html').read_text(encoding='utf-8');parser=Inputs();parser.feed(source)
        self.assertTrue(set(KEYS)<=set(ROUND25_OPTIONS))
        actual={name.removeprefix('r25_') for name in parser.inputs if name.startswith('r25_')}
        self.assertEqual(actual,set(KEYS))
        for key in KEYS:
            rows=parser.inputs['r25_'+key];self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['type'],'checkbox');self.assertNotIn('checked',rows[0])
            self.assertFalse(normalize_options()[key]);self.assertIn('aria-describedby',rows[0])
        self.assertIn('verified native SAT clause subset',source)
        self.assertIn('not full solver state',source)

    def node(self,body):
        if not shutil.which('node'):self.skipTest('Node is needed to execute the dashboard JavaScript')
        result=subprocess.run([shutil.which('node'),'-e',HARNESS+body,str(ROOT/'Dashboard.html')],
                              cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    def test_actual_js_dependency_normalization_matches_backend(self):
        cases=[{},*({key:True} for key in KEYS),{'mixed_proof_dag':True,'proof_transport':False,'smart_learning':False},
               {'native_explanations':True,'native_search':False,'lcg':False},
               {'sat_core_lifting':True,'persistent_sat':False},
               {'native_sat_persistence':True,'persistent_sat':False},
               {'conflict_pb':True,'pb_learning':False},
               {'hall_rematch':True,'incremental_hall':False},
               {'reversible_gold_graph':True,'strand_connectivity':False,'biconnected':False},
               {'integer_lattice':True,'affine_reasoning':False,'incremental_affine':False},
               {'durable_regions':True,'incremental_regions':False},
               {'gold_elimination':True,'compact_factors':False},
               {'symbolic_coverage':True,'shared_coverage':False},
               {'witness_pool':True,'sat_assist':False}]
        code='const cases='+json.dumps(cases)+';\n'
        code+="console.log(JSON.stringify(cases.map(item=>run('normalizeSearchOptions('+JSON.stringify(item)+')'))));"
        observed=json.loads(self.node(code))
        for given,actual in zip(cases,observed):
            expected=normalize_options(given)
            self.assertEqual(actual,{key:expected[key] for key in actual})
        defaults=json.loads(self.node('console.log(JSON.stringify(run("readSearchOptions()")));'))
        for key in KEYS:self.assertFalse(defaults[key])

    def test_algorithm_scope_and_posted_dependencies(self):
        self.node(r"""
assert.equal(element('r25_sat_core_lifting').disabled,true);
select('sat');assert.equal(element('r25_sat_core_lifting').disabled,false);
assert.equal(element('r25_symbolic_coverage').disabled,true);
edit('sat_core_lifting',true);
assert.equal(element('r24_persistent_sat').checked,true);
assert.equal(element('r25_symbolic_coverage').disabled,false);
edit('symbolic_coverage',true);
assert.equal(element('r24_shared_coverage').checked,true);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.algorithm,'sat');assert.equal(sent.seconds,0);
for(const key of ['sat_core_lifting','symbolic_coverage','persistent_sat','shared_coverage'])assert.equal(sent.search_options[key],true,key);
idle();select('dfs');
assert.equal(element('r25_sat_core_lifting').checked,true,'Preserve a saved incompatible preference');
assert.equal(element('r25_sat_core_lifting').disabled,false,'Allow clearing it');
assert.equal(element('startBtn').disabled,true);
const before=context.fetchCalls.length;run("command('start')");assert.equal(context.fetchCalls.length,before,'Do not post a scope-invalid option');
edit('sat_core_lifting',false);
assert.equal(element('r25_sat_core_lifting').disabled,true);
select('hybrid');assert.equal(element('startBtn').disabled,true,'Symbolic coverage has no hybrid ledger hook');
edit('symbolic_coverage',false);assert.equal(element('r25_symbolic_coverage').disabled,true);
edit('mixed_proof_dag',true);
for(const id of ['r25_resumable_proof_checking','r24_proof_transport','r23_smart_learning'])assert.equal(element(id).checked,true,id);
select('gpu-domains');assert.equal(element('startBtn').disabled,true,'GPU domains do not run this CPU proof DAG');
""")

    def test_native_subset_capability_dependencies_and_clearable_saved_choices(self):
        self.node(r"""
select('sat');assert.equal(element('r25_native_sat_persistence').disabled,true,'Require a server capability result');
run("state.capabilities={python:'3.12',algorithms:{sat:{available:true},dfs:{available:true},'cp-sat':{available:true}},backends:{},options:{native_sat_persistence:{available:true,reason:''}},features:{}};renderState();");
assert.equal(element('r25_native_sat_persistence').disabled,false);
edit('native_sat_persistence',true);
assert.equal(element('r24_persistent_sat').checked,true);
assert.equal(element('startBtn').disabled,false);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");
let sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.search_options.native_sat_persistence,true);assert.equal(sent.search_options.persistent_sat,true);
idle();select('sat');
run("state.capabilities={python:'3.15',algorithms:{sat:{available:true},dfs:{available:true},'cp-sat':{available:true}},backends:{},options:{native_sat_persistence:{available:false,reason:'Proof logging is unavailable on Windows'}},features:{}};renderState();");
assert.equal(element('r25_native_sat_persistence').checked,true);
assert.equal(element('r25_native_sat_persistence').disabled,false,'Unsupported saved flags must be clearable');
assert.equal(element('startBtn').disabled,true);assert.match(element('runtimeHint').textContent,/Windows/);
const before=context.fetchCalls.length;run("command('start')");assert.equal(context.fetchCalls.length,before);
select('cp-sat');
assert.equal(element('advancedOptions').classList.contains('hidden'),false,'Show saved incompatible controls so they can be cleared');
assert.equal(element('r25_native_sat_persistence').disabled,false,'Keep the control clearable even outside exact-search modes');
assert.equal(element('startBtn').disabled,true);
edit('native_sat_persistence',false);assert.equal(element('r25_native_sat_persistence').disabled,true);
assert.equal(element('advancedOptions').classList.contains('hidden'),true);
select('sat');assert.equal(element('startBtn').disabled,false,'Ordinary persistent SAT can import a checked subset without enabling a logger');
assert.equal(run('readSearchOptions().native_sat_persistence'),false);
""")

    def test_weighted_conflicts_and_integer_lattice_scope_and_posted_options(self):
        self.node(r"""
for(const algorithm of ['dfs','parallel-dfs','hybrid']){
 select(algorithm);
 for(const key of ['conflict_pb','integer_lattice'])assert.equal(element('r25_'+key).disabled,false);
}
edit('conflict_pb',true);edit('integer_lattice',true);
assert.equal(element('r24_pb_learning').checked,true);
assert.equal(element('r24_affine_reasoning').checked,false,'Integer lattice has no affine prerequisite');
assert.equal(element('r25_incremental_affine').checked,false);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.algorithm,'hybrid');
for(const key of ['conflict_pb','pb_learning','integer_lattice'])assert.equal(sent.search_options[key],true,key);
idle();select('sat');
assert.equal(element('startBtn').disabled,true);
for(const key of ['conflict_pb','integer_lattice']){
 assert.equal(element('r25_'+key).disabled,false,'Clear saved DFS-only preference');
 edit(key,false);assert.equal(element('r25_'+key).disabled,true);
}
assert.equal(element('startBtn').disabled,false);
""")

    def test_matching_and_gold_graph_reuse_dependencies_and_scope(self):
        self.node(r"""
for(const algorithm of ['dfs','parallel-dfs','hybrid']){
 select(algorithm);
 for(const key of ['hall_rematch','reversible_gold_graph'])assert.equal(element('r25_'+key).disabled,false);
}
element('strandInput').checked=false;element('strandInput').listeners.change();
element('biconnectedInput').checked=false;element('biconnectedInput').listeners.change();
edit('hall_rematch',true);edit('reversible_gold_graph',true);
for(const id of ['r24_incremental_hall','strandInput','biconnectedInput'])assert.equal(element(id).checked,true,id);
select('parallel-dfs');
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.algorithm,'parallel-dfs');
for(const key of ['hall_rematch','incremental_hall','reversible_gold_graph','strand_connectivity','biconnected'])assert.equal(sent.search_options[key],true,key);
idle();select('sat');assert.equal(element('startBtn').disabled,true);
for(const key of ['hall_rematch','reversible_gold_graph']){
 assert.equal(element('r25_'+key).disabled,false,'Saved incompatible choice remains clearable');
 edit(key,false);assert.equal(element('r25_'+key).disabled,true);
}
assert.equal(element('startBtn').disabled,false);
""")

    def test_regional_options_dependencies_scope_and_save_failure_stats(self):
        self.node(r"""
for(const algorithm of ['dfs','parallel-dfs','hybrid']){
 select(algorithm);
 for(const key of ['incremental_regions','durable_regions','compact_factors','gold_elimination'])assert.equal(element('r25_'+key).disabled,false);
}
edit('durable_regions',true);edit('gold_elimination',true);
assert.equal(element('r25_incremental_regions').checked,true);
assert.equal(element('r25_compact_factors').checked,true);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
for(const key of ['durable_regions','incremental_regions','compact_factors','gold_elimination'])assert.equal(sent.search_options[key],true,key);
idle();select('sat');assert.equal(element('startBtn').disabled,true);
for(const key of ['durable_regions','incremental_regions','gold_elimination','compact_factors'])edit(key,false);
select('parallel-dfs');
run("state.status={state:'stopped',method:'constraint',algorithm:'parallel-dfs',stats:{region_restrict_reuses:7,region_cursor_writes:4,region_cursor_loads:3,region_cursor_write_failures:2,compact_factor_completed:1,regional_solutions_validated:1}};renderState();");
for(const [id,value] of [['r25RegionReuse','7'],['r25RegionWrites','4'],['r25RegionLoads','3'],['r25RegionWriteFailures','2'],['r25WholeComplete','1'],['r25GoldAccepted','1']])assert.equal(element(id).textContent,value,id);
assert.match(element('smartSearchHint').textContent,/could not be saved/);
""")

    def test_new_regional_stages_are_independent_scoped_and_report_reuse(self):
        self.node(r"""
const keys=['separator_search','selective_refinement','joint_symbolic_regions','algebraic_connectivity','residual_isomorphism','gold_pricing'];
for(const algorithm of ['dfs','parallel-dfs','hybrid']){
 select(algorithm);for(const key of keys)assert.equal(element('r25_'+key).disabled,false,key);
}
for(const key of keys)edit(key,true);
for(const key of ['elimination','relaxed_diagrams','representative_connectivity','residual_reuse'])assert.equal(element('r24_'+key).checked,false,key);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
for(const key of keys)assert.equal(sent.search_options[key],true,key);
idle();select('sat');assert.equal(element('startBtn').disabled,true);
for(const key of keys)edit(key,false);
select('parallel-dfs');
run("state.status={state:'stopped',method:'constraint',algorithm:'parallel-dfs',stats:{andor_component_reuses:7,selective_dd_reused_layers:9,joint_region_cache_hits:3,algebraic_connectivity_basis_hits:4,isomorphic_hits:2,gold_pricing_failures:1}};renderState();");
for(const[id,value]of [['r25SeparatorReuse','7'],['r25RefinementReuse','9'],['r25JointReuse','3'],['r25BasisReuse','4'],['r25EquivalentReuse','2'],['r25PricingFailures','1']])assert.equal(element(id).textContent,value,id);
run("state.capabilities={algorithms:{'parallel-dfs':{available:true}},options:{gold_pricing:{available:false,reason:'Requires OR-Tools in this Python environment.'}},features:{},backends:{}};renderState();");
assert.equal(element('r25_gold_pricing').disabled,true);
edit('gold_pricing',true);assert.equal(element('r25_gold_pricing').disabled,false);assert.equal(element('startBtn').disabled,true);
assert.match(element('runtimeHint').textContent,/OR-Tools/);
const before=context.fetchCalls.length;run("command('start')");assert.equal(context.fetchCalls.length,before);
edit('gold_pricing',false);assert.equal(element('startBtn').disabled,false);
""")

    def test_joint_chain_and_completed_certificate_controls_and_counters(self):
        self.node(r"""
select('parallel-dfs');edit('multichain_routing',true);edit('region_certificates',true);
for(const id of ['r25_durable_regions','r23_extended_regions'])assert.equal(element(id).checked,false,id);
element('seconds').value='0';element('replicaInput').value='128';element('seedInput').value='17';
run("command('start')");const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.search_options.multichain_routing,true);assert.equal(sent.search_options.region_certificates,true);
idle();select('gpu-domains');assert.equal(element('startBtn').disabled,true);
edit('multichain_routing',false);edit('region_certificates',false);select('parallel-dfs');
run("state.status={state:'stopped',method:'constraint',algorithm:'parallel-dfs',stats:{multichain_failures:1,region_certificate_saves:2,region_certificate_loads:3,region_certificate_verified_transitions:889,region_certificate_rejected:4,region_certificate_write_failures:1}};renderState();");
for(const[id,value]of [['r25JointChainFailures','1'],['r25CertificateSaves','2'],['r25CertificateLoads','3'],['r25CertificateTransitions','889'],['r25CertificateRejected','4'],['r25CertificateWriteFailures','1']])assert.equal(element(id).textContent,value,id);
assert.match(element('smartSearchHint').textContent,/could not be saved/);
""")

    def test_missing_native_build_blocks_start_but_saved_choice_is_clearable(self):
        self.node(r"""
run("state.capabilities={algorithms:{dfs:{available:true}},options:{native_explanations:{available:false,reason:'Run python build_native.py first.'}},features:{},backends:{}};renderState();");
assert.equal(element('r25_native_explanations').disabled,true);
edit('native_explanations',true);
assert.equal(element('r25_native_explanations').disabled,false);
assert.equal(element('startBtn').disabled,true);assert.match(element('runtimeHint').textContent,/build_native/);
const before=context.fetchCalls.length;run("command('start')");assert.equal(context.fetchCalls.length,before);
edit('native_explanations',false);assert.equal(element('startBtn').disabled,false);
""")

    def test_capability_gating_native_core_and_active_preferences(self):
        self.node(r"""
run("state.capabilities={python:'3.15',algorithms:{dfs:{available:true}},backends:{auto:{available:true}},options:{sat_assist:{available:false,reason:'Requires python-sat'},shared_sat_assistant:{available:false,reason:'Requires python-sat'},smart_learning:{available:false,reason:'Requires python-sat'}},features:{}};renderState();");
assert.equal(element('r25_shared_sat_assistant').disabled,true);
edit('shared_sat_assistant',true);
assert.equal(element('satAssistInput').checked,true);
assert.equal(element('r25_shared_sat_assistant').disabled,false);
assert.equal(element('startBtn').disabled,true);assert.match(element('runtimeHint').textContent,/python-sat/);
edit('shared_sat_assistant',false);element('satAssistInput').checked=false;element('satAssistInput').listeners.change();
edit('mixed_proof_dag',true);
assert.equal(element('startBtn').disabled,false,'Independent proof checking works without python-sat');
element('nativeInput').value='python';element('nativeInput').listeners.change();
assert.equal(element('r25_native_explanations').disabled,true);
edit('native_explanations',true);assert.equal(element('startBtn').disabled,true);
assert.match(element('runtimeHint').textContent,/automatic or compiled/);
element('nativeInput').value='auto';element('nativeInput').listeners.change();
assert.equal(element('startBtn').disabled,false);
const preferences=JSON.stringify(storage.values);
run("state.process_running=true;state.status={algorithm:'dfs',method:'constraint',state:'running',search_options:{native_explanations:false,mixed_proof_dag:false,resumable_proof_checking:false,shared_sat_assistant:false,sat_core_lifting:false,symbolic_coverage:false,native:'compiled'}};renderAlgorithmControls();");
for(const key of run('round25Keys'))assert.equal(element('r25_'+key).disabled,true);
assert.equal(element('r25_native_explanations').checked,false);assert.equal(JSON.stringify(storage.values),preferences);
run("state.process_running=false;renderAlgorithmControls();");
assert.equal(element('r25_native_explanations').checked,true);assert.equal(element('r25_mixed_proof_dag').checked,true);
assert.equal(JSON.stringify(storage.values),preferences);
""")


if __name__=='__main__':unittest.main()
