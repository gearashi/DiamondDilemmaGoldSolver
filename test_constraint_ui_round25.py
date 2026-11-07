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
      'multichain_routing','region_certificates','gpu_joint_support','gpu_propagated_forks','gpu_event_propagation','gpu_precise_reasons','gpu_shared_clauses','gpu_lane_balance','gpu_incremental_checkpoint','durable_hybrid','obstruction_chains','cycle_crossover','lazy_gold_scoring','adaptive_temperatures')
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

    def test_cycle_crossover_is_posted_only_for_hybrid(self):
        self.node(r"""
select('dfs');assert.equal(element('r25_cycle_crossover').disabled,true);
select('hybrid');edit('cycle_crossover',true);
assert.equal(run('readSearchOptions()').cycle_crossover,true);
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
assert.equal(JSON.parse(context.fetchCalls.at(-1)[1].body).search_options.cycle_crossover,true);
idle();select('parallel-dfs');assert.equal(element('startBtn').disabled,true);
edit('cycle_crossover',false);assert.equal(element('r25_cycle_crossover').disabled,true);
""")

    def test_cycle_crossover_work_counters_are_distinct(self):
        self.node(r"""
run(`state={process_running:true,status:{state:'running',method:'constraint',algorithm:'hybrid',gpu_sampling:{sampling_experiments:{"cycle_crossover_rounds": 1, "cycle_parent_pairs": 2, "cycle_components": 3, "cycle_candidates": 4, "cycle_duplicate_skips": 5, "cycle_domain_rejections": 6, "cycle_improvements": 7, "cycle_gold_solutions": 8, "cycle_history_evictions": 9, "cycle_injections": 10, "cycle_parent_scores": 11, "cycle_parent_duplicates": 12, "cycle_pair_cursor_evictions": 13, "cycle_offers": 14}}},history:[]};renderState();`);
for(const[id,n]of [["r25CycleRounds", 1], ["r25CyclePairs", 2], ["r25CycleComponents", 3], ["r25CycleCandidates", 4], ["r25CycleDuplicates", 5], ["r25CycleDomains", 6], ["r25CycleImprovements", 7], ["r25CycleGold", 8], ["r25CycleEvictions", 9], ["r25CycleInjections", 10], ["r25CycleParentScores", 11], ["r25CycleParentDuplicates", 12], ["r25CycleCursorEvictions", 13], ["r25CycleOffers", 14]])assert.equal(element(id).textContent,String(n));
""")

    def test_lazy_gold_scoring_scope_and_dependencies(self):
        self.node(r"""
select('dfs');assert.equal(element('r25_lazy_gold_scoring').disabled,true);
select('hybrid');edit('lazy_gold_scoring',true);
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const opts=JSON.parse(context.fetchCalls.at(-1)[1].body).search_options;
assert.equal(opts.lazy_gold_scoring,true);assert.equal(opts.gpu_gold_sampling,true);assert.equal(opts.ejection_chains,true);
idle();select('parallel-dfs');assert.equal(element('startBtn').disabled,true);
edit('lazy_gold_scoring',false);assert.equal(element('r25_lazy_gold_scoring').disabled,true);
""")

    def test_lazy_gold_scoring_work_counters(self):
        self.node(r"""
run(`state={process_running:true,status:{state:'running',method:'constraint',algorithm:'hybrid',gpu_sampling:{sampling_experiments:{"gold_old_base_scans_saved": 1, "gold_old_loop_traces_saved": 2, "gold_candidate_loop_traces": 3, "gold_tabu_traces_skipped": 4, "gold_metropolis_traces_skipped": 5, "gold_early_metropolis_draws": 6}}},history:[]};renderState();`);
for(const[id,n]of [["r25GoldBaseSaved", 1], ["r25GoldOldTracesSaved", 2], ["r25GoldCandidateTraces", 3], ["r25GoldTabuSkipped", 4], ["r25GoldMetropolisSkipped", 5], ["r25GoldEarlyDraws", 6]])assert.equal(element(id).textContent,String(n));
""")

    def test_adaptive_temperature_scope_and_exchange_dependency(self):
        self.node(r"""
select('dfs');assert.equal(element('r25_adaptive_temperatures').disabled,true);
select('hybrid');edit('adaptive_temperatures',true);
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const opts=JSON.parse(context.fetchCalls.at(-1)[1].body).search_options;
assert.equal(opts.adaptive_temperatures,true);assert.equal(opts.replica_exchange,true);
idle();select('parallel-dfs');assert.equal(element('startBtn').disabled,true);
edit('adaptive_temperatures',false);assert.equal(element('r25_adaptive_temperatures').disabled,true);
""")

    def test_adaptive_temperature_feedback_counters(self):
        self.node(r"""
run(`state={process_running:true,status:{state:'running',method:'constraint',algorithm:'hybrid',gpu_sampling:{sampling_experiments:{"temperature_feedback_windows": 1, "temperature_eligible_windows": 2, "temperature_adjustments": 3, "temperature_empty_windows": 4, "temperature_freezes": 1, "temperature_feedback_attempts": 6, "temperature_feedback_accepts": 7}}},history:[]};renderState();`);
for(const[id,n]of [["r25TemperatureWindows", 1], ["r25TemperatureEligible", 2], ["r25TemperatureAdjustments", 3], ["r25TemperatureEmpty", 4], ["r25TemperatureFrozen", 1], ["r25TemperatureAttempts", 6], ["r25TemperatureAccepts", 7]])assert.equal(element(id).textContent,String(n));
""")

    def node(self,body):
        if not shutil.which('node'):self.skipTest('Node is needed to execute the dashboard JavaScript')
        result=subprocess.run([shutil.which('node'),'-e',HARNESS+body,str(ROOT/'Dashboard.html')],
                              cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    def test_obstruction_chain_scope_dependency_and_work_counters(self):
        self.node(r"""
assert.equal(element('r25_obstruction_chains').disabled,true);
select('hybrid');edit('obstruction_chains',true);
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const opts=JSON.parse(context.fetchCalls.at(-1)[1].body).search_options;
assert.equal(opts.obstruction_chains,true);assert.equal(opts.ejection_chains,true);
assert.equal(opts.durable_hybrid,false,'Guided moves do not require checkpoint continuation');
idle();select('dfs');assert.equal(element('startBtn').disabled,true);edit('obstruction_chains',false);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'hybrid',gpu_sampling:{sampling_experiments:{guided_proposals:7,guided_candidate_checks:900,guided_obstruction_starts:6,guided_base_improvements:3}}},history:[]};renderState();");
for(const[id,n]of [['r25GuidedProposals',7],['r25GuidedChecks',900],['r25GuidedStarts',6],['r25GuidedImprovements',3]])assert.equal(element(id).textContent,String(n));
""")

    def test_durable_hybrid_control_and_observation_counters(self):
        self.node(r"""
assert.equal(element('r25_durable_hybrid').disabled,true);
select('hybrid');edit('durable_hybrid',true);
assert.ok(element('algorithmHint').textContent.includes('Saves GPU sampling'));
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.algorithm,'hybrid');assert.equal(sent.search_options.durable_hybrid,true);
idle();select('dfs');assert.equal(element('startBtn').disabled,true);
assert.equal(element('r25_durable_hybrid').disabled,false);edit('durable_hybrid',false);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'hybrid',gpu_sampling:{sampler_restores:2,move_proposals:4096,archive_checks:80,archive_repeated_observations:19}},history:[]};renderState();");
for(const[id,n]of [['r25SamplerRestores',2],['r25SamplerMoves',4096],['r25SamplerArchiveChecks',80],['r25SamplerArchiveRepeats',19]])assert.equal(element(id).textContent,Number(n).toLocaleString('en-US'));
""")

    def test_gpu_joint_support_scope_posted_option_and_actual_counters(self):
        self.node(r"""
assert.equal(element('r25_gpu_joint_support').disabled,true);
select('gpu-domains');assert.equal(element('r25_gpu_joint_support').disabled,false);
edit('gpu_joint_support',true);
element('seconds').value='0';element('seedInput').value='17';
run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(sent.algorithm,'gpu-domains');assert.equal(sent.search_options.gpu_joint_support,true);
idle();select('dfs');assert.equal(element('r25_gpu_joint_support').checked,true);
assert.equal(element('r25_gpu_joint_support').disabled,false);assert.equal(element('startBtn').disabled,true);
edit('gpu_joint_support',false);assert.equal(element('r25_gpu_joint_support').disabled,true);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',stats:{joint_support_checks:53,joint_exclusions:4,joint_subset_checks:61,joint_word_scans:82,joint_tile_visits:93}},history:[]};renderState();");
for(const [id,n]of [['r25GPUJointChecks',53],['r25GPUJointExclusions',4],['r25GPUJointSubsets',61],['r25GPUJointWords',82],['r25GPUJointTiles',93]])assert.equal(element(id).textContent,String(n),id);
""")

    def test_gpu_parent_forks_control_and_visible_preparation(self):
        self.node(r"""
assert.equal(element('r25_gpu_propagated_forks').disabled,true);
select('gpu-domains');assert.equal(element('r25_gpu_propagated_forks').disabled,false);
edit('gpu_propagated_forks',true);
assert.equal(element('r25_gpu_joint_support').checked,false,'Parent reuse is independent of joint support');
select('hybrid');assert.equal(element('startBtn').disabled,true);
assert.equal(element('r25_gpu_propagated_forks').disabled,false);edit('gpu_propagated_forks',false);
run("state={process_running:true,control_token:'test',status:{state:'running',method:'constraint',algorithm:'gpu-domains',phase_message:'Searching',stats:{gpu_domain_preparing:1,fork_parents:2,fork_children:8,fork_root_removals:11,fork_root_checks:23,fork_reused_assignments:32,fork_reused_unions:48}},history:[]};renderState();");
assert.equal(element('runStatus').textContent,'Preparing shared GPU parent');
assert.equal(element('stopBtn').disabled,false,'Stop remains available during parent preparation');
assert.ok(element('replicaNote').textContent.includes('Preparing shared parent'));
for(const [id,n]of [['r25GPUForkParents',2],['r25GPUForkChildren',8],['r25GPUForkRemovals',11],['r25GPUForkChecks',23],['r25GPUForkAssignments',32],['r25GPUForkUnions',48]])assert.equal(element(id).textContent,String(n),id);
run("state.stop_requested=true;state.display_state='stopping';renderState();");
assert.equal(element('runStatus').textContent,'Stopping','Stop status takes priority');
""")

    def test_gpu_event_control_has_independent_scope_and_separate_work_counters(self):
        self.node(r"""
assert.equal(element('r25_gpu_event_propagation').disabled,true);
select('gpu-domains');edit('gpu_event_propagation',true);
for(const key of ['gpu_joint_support','gpu_propagated_forks'])assert.equal(element('r25_'+key).checked,false,'Event propagation is independent');
select('sat');assert.equal(element('startBtn').disabled,true);assert.equal(element('r25_gpu_event_propagation').disabled,false);
edit('gpu_event_propagation',false);assert.equal(element('r25_gpu_event_propagation').disabled,true);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',stats:{event_enqueues:19,event_cell_scans:17,event_inventory_checks:7,event_queue_word_scans:23,event_inventory_words:11}},history:[]};renderState();");
for(const [id,n]of [['r25GPUEventEnqueues',19],['r25GPUEventScans',17],['r25GPUEventInventory',7],['r25GPUEventQueueWords',23],['r25GPUEventInventoryWords',11]])assert.equal(element(id).textContent,String(n),id);
""")

    def test_gpu_precise_reasons_independent_control_and_cache_counters(self):
        self.node(r"""
assert.equal(element('r25_gpu_precise_reasons').disabled,true);
select('gpu-domains');edit('gpu_precise_reasons',true);
for(const key of ['gpu_joint_support','gpu_propagated_forks','gpu_event_propagation'])assert.equal(element('r25_'+key).checked,false);
assert.equal(element('replicaInput').max,'512','Bounded cache still supports this lane cap');
select('dfs');assert.equal(element('startBtn').disabled,true);assert.equal(element('r25_gpu_precise_reasons').disabled,false);
edit('gpu_precise_reasons',false);assert.equal(element('r25_gpu_precise_reasons').disabled,true);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',stats:{precise_probes:100,precise_hits:40,precise_fallbacks:3,precise_stores:70,precise_invalidations:9,precise_support_words:250,precise_support_values:300,precise_dropped:2}},history:[]};renderState();");
for(const [id,n]of [['r25GPUReasonProbes',100],['r25GPUReasonHits',40],['r25GPUReasonFallbacks',3],['r25GPUReasonStores',70],['r25GPUReasonInvalidations',9],['r25GPUReasonWords',250],['r25GPUReasonValues',300],['r25GPUReasonDropped',2]])assert.equal(element(id).textContent,String(n),id);
""")

    def test_gpu_sharing_requires_verifier_but_pending_counts_are_not_proofs(self):
        self.node(r"""
select('gpu-domains');
run("state.capabilities={options:{gpu_shared_clauses:{available:false,reason:'Requires python-sat.'}}};renderState();");
assert.equal(element('r25_gpu_shared_clauses').disabled,true);
edit('gpu_shared_clauses',true);assert.equal(element('r25_gpu_shared_clauses').disabled,false);
assert.equal(element('startBtn').disabled,true);assert.ok(element('runtimeHint').textContent.includes('python-sat'));
edit('gpu_shared_clauses',false);
run("state.capabilities.options.gpu_shared_clauses.available=true;renderState();");
edit('gpu_shared_clauses',true);assert.equal(element('sharedConflictsInput').checked,false,'Separate GPU scope flag');
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);assert.equal(sent.search_options.gpu_shared_clauses,true);
idle();select('dfs');assert.equal(element('startBtn').disabled,true);edit('gpu_shared_clauses',false);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',stats:{gpu_claims_harvested:9,gpu_claims_offered:8,gpu_claims_oversize:1,gpu_shared_imports:3,gpu_shared_comparisons:21,gpu_shared_verification_calls:4,gpu_shared_verified:2,gpu_shared_rejected:1,gpu_shared_unknown:1,gpu_shared_entries:2,gpu_shared_pending:5,gpu_shared_verifier_clauses:88,gpu_verified_root_unsat:0}},history:[]};renderState();");
for(const [id,n]of [['r25GPUClaimsHarvested',9],['r25GPUClaimsOffered',8],['r25GPUClaimsOversize',1],['r25GPUSharedImports',3],['r25GPUSharedComparisons',21],['r25GPUVerificationCalls',4],['r25GPUSharedVerified',2],['r25GPUSharedRejected',1],['r25GPUSharedUnknown',1],['r25GPUSharedEntries',2],['r25GPUSharedPending',5],['r25GPUVerifierClauses',88],['r25GPUVerifiedRoot',0]])assert.equal(element(id).textContent,String(n),id);
""")

    def test_gpu_lane_balance_preserves_scope_and_exposes_rebuild_work(self):
        self.node(r"""
assert.equal(element('r25_gpu_lane_balance').disabled,true);
select('gpu-domains');edit('gpu_lane_balance',true);
assert.equal(element('r25_gpu_propagated_forks').checked,false,'Refill does not require parent forks');
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
const sent=JSON.parse(context.fetchCalls.at(-1)[1].body);assert.equal(sent.search_options.gpu_lane_balance,true);
idle();select('hybrid');assert.equal(element('startBtn').disabled,true);assert.equal(element('r25_gpu_lane_balance').disabled,false);
edit('gpu_lane_balance',false);assert.equal(element('r25_gpu_lane_balance').disabled,true);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',stats:{\"gpu_lane_refills\": 2, \"gpu_lane_donations\": 3, \"gpu_lane_rebuilt_roots\": 4, \"gpu_lane_discarded_frames\": 5, \"gpu_lane_discarded_removals\": 6, \"gpu_lane_owners_completed\": 7, \"gpu_lane_donation_cubes\": 8, \"gpu_lane_balance_deferrals\": 9, \"gpu_lane_idle\": 10, \"gpu_lane_owner_groups\": 11, \"gpu_lane_queue_peak\": 12}},history:[]};renderState();");
for(const[id,n]of [["r25GPULaneRefills", 2], ["r25GPULaneDonations", 3], ["r25GPULaneRebuiltRoots", 4], ["r25GPULaneDiscardedFrames", 5], ["r25GPULaneDiscardedRemovals", 6], ["r25GPULaneOwnersCompleted", 7], ["r25GPULaneDonationCubes", 8], ["r25GPULaneBalanceDeferrals", 9], ["r25GPULaneIdle", 10], ["r25GPULaneOwnerGroups", 11], ["r25GPULaneQueuePeak", 12]])assert.equal(element(id).textContent,String(n),id);
""")

    def test_gpu_checkpoint_scope_sizes_and_fallback_reason(self):
        self.node(r"""
assert.equal(element('r25_gpu_incremental_checkpoint').disabled,true);
select('gpu-domains');edit('gpu_incremental_checkpoint',true);
assert.equal(element('r25_gpu_lane_balance').checked,false,'Checkpoint reuse has no balancing dependency');
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
assert.equal(JSON.parse(context.fetchCalls.at(-1)[1].body).search_options.gpu_incremental_checkpoint,true);
idle();select('dfs');assert.equal(element('startBtn').disabled,true);edit('gpu_incremental_checkpoint',false);
run("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'gpu-domains',gpu_checkpoint_status:'legacy_fallback',gpu_checkpoint_note:'Retention limit reached',stats:{\"gpu_checkpoint_host_copies\": 20, \"gpu_checkpoint_reused_chunks\": 21, \"gpu_checkpoint_fallbacks\": 22, \"gpu_checkpoint_rollbacks\": 23, \"gpu_checkpoint_packed_bytes\": 24, \"gpu_checkpoint_compressed_bytes\": 25, \"gpu_checkpoint_omitted_bytes\": 26, \"gpu_checkpoint_store_bytes\": 27}},history:[]};renderState();");
for(const[id,n]of [["r25GPUCheckpointCopies", 20], ["r25GPUCheckpointReuses", 21], ["r25GPUCheckpointFallbacks", 22], ["r25GPUCheckpointRollbacks", 23], ["r25GPUCheckpointPacked", 24], ["r25GPUCheckpointCompressed", 25], ["r25GPUCheckpointOmitted", 26], ["r25GPUCheckpointStore", 27]])assert.equal(element(id).textContent,String(n),id);
assert.equal(element('r25GPUCheckpointStatus').textContent,'Self-contained fallback');
assert.equal(element('r25GPUCheckpointNote').textContent,'Retention limit reached');
""")

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
// CP-SAT now has its own regional compiler control in this panel.
assert.equal(element('advancedOptions').classList.contains('hidden'),false);
assert.equal(element('r26_regional_compiler').disabled,false);
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
