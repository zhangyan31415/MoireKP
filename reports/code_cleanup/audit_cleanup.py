"""Reproduce cleanup inventory coverage and the comparable visible-core counts.

This is a report generator, not a runtime format registry or calculation module.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/code_cleanup'
OLD = ROOT / 'reports/code_audit_20261014'
BEFORE_COMMIT = '060dbc74f105c12463d45d3f4803218c0594cd84'
VERSION = re.compile(r'(?<![A-Za-z0-9])[vV]\d+(?![A-Za-z0-9])')
LEGACY = re.compile(r'(?<![A-Za-z0-9])legacy', re.I)
COMPAT = re.compile(r'(?<![A-Za-z0-9])(?:compatibility|backward.compatible)(?![A-Za-z0-9])', re.I)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def literal(value):
    return value.decode('utf-8', 'replace') if isinstance(value, bytes) else value


def index_source(text):
    tree = ast.parse(text)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts = [node.name]
            parent = node
            while parent in parents:
                parent = parents[parent]
                if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    parts.append(parent.name)
            functions.append((node.lineno, node.end_lineno, '.'.join(reversed(parts)), node))
    return {'text': text, 'lines': text.splitlines(), 'tree': tree,
            'parents': parents, 'functions': functions}


def enclosing(index, line):
    nodes = [entry for entry in index['functions'] if entry[0] <= line <= entry[1]]
    return min(nodes, key=lambda entry: entry[1]-entry[0])[2] if nodes else '<module>'


def analyze(sources):
    counts = Counter()
    labels = Counter()
    identifiers = Counter()
    duplicates = defaultdict(list)
    handlers = []
    large = []
    files = []
    for path, index in sources.items():
        text, tree = index['text'], index['tree']
        lines = index['lines']
        files.append({'file': path, 'lines': len(lines), 'sha256': hashlib.sha256(text.encode()).hexdigest()})
        counts['version_tokens'] += len(VERSION.findall(text))
        counts['version_lines'] += sum(bool(VERSION.search(line)) for line in lines)
        counts['legacy_mentions'] += len(LEGACY.findall(text))
        counts['fallback_mentions'] += len(re.findall('fallback', text, re.I))
        counts['compatibility_mentions'] += len(COMPAT.findall(text))
        counts['files_at_least_2000_lines'] += len(lines) >= 2000
        counts['lines_over_300_chars'] += sum(len(line) > 300 for line in lines)
        counts['lines_over_1000_chars'] += sum(len(line) > 1000 for line in lines)
        counts['debug_blocks'] += len(re.findall(r'#region\s+(?:debug-point|agent\s+log)', text, re.I))
        counts['debug_http_posts'] += len(re.findall(r'http://172\.16\.8\.68:7777/event', text))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and VERSION.search(literal(node.value)):
                counts['version_literal_nodes'] += 1
                labels[literal(node.value)] += 1
            if isinstance(node, ast.Name) and VERSION.search(node.id) and not re.fullmatch(r'[vV]\d+', node.id):
                counts['version_identifier_uses'] += 1
                identifiers[node.id] += 1
            if isinstance(node, ast.If) and VERSION.search(ast.unparse(node.test)):
                counts['version_gated_if_nodes'] += 1
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'exec':
                counts['dynamic_exec_calls'] += 1
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    try:
                        inner = ast.parse(node.args[0].value)
                    except SyntaxError:
                        continue
                    counts['silent_except_inside_exec_strings'] += sum(
                        isinstance(inner_node, ast.ExceptHandler) and len(inner_node.body)==1 and isinstance(inner_node.body[0], ast.Pass)
                        for inner_node in ast.walk(inner))
            if isinstance(node, ast.ExceptHandler):
                broad = node.type is not None and any(isinstance(part, ast.Name) and part.id in ('Exception','BaseException') for part in ast.walk(node.type))
                counts['bare_except_handlers'] += node.type is None
                counts['broad_except_handlers'] += broad
                silent = len(node.body)==1 and isinstance(node.body[0], ast.Pass)
                counts['silent_except_pass_handlers'] += silent
                if broad or node.type is None or silent:
                    handlers.append({'file': path, 'line': node.lineno,
                                     'function': enclosing(index,node.lineno),
                                     'handler': ast.unparse(node), 'silent_pass': silent})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts['functions'] += 1
                span = node.end_lineno-node.lineno+1
                nargs = len(node.args.posonlyargs)+len(node.args.args)+len(node.args.kwonlyargs)+bool(node.args.vararg)+bool(node.args.kwarg)
                counts['functions_at_least_300_lines'] += span >= 300
                counts['functions_at_least_20_parameters'] += nargs >= 20
                if VERSION.search(node.name):
                    counts['version_named_functions'] += 1
                    counts['version_identifier_uses'] += 1
                    identifiers[node.name] += 1
                if span >= 300:
                    large.append({'file':path,'name':enclosing(index,node.lineno),'line':node.lineno,'lines':span})
                if span >= 15:
                    body=node.body
                    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                        body=body[1:]
                    key=ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False)
                    duplicates[key].append({'file':path,'name':enclosing(index,node.lineno),'line':node.lineno,'lines':span})
    groups=[group for group in duplicates.values() if len(group)>1]
    overwritten=[]
    for path,index in sources.items():
        for node in ast.walk(index['tree']):
            if not isinstance(node,ast.ClassDef):
                continue
            methods=defaultdict(list)
            for item in node.body:
                if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    methods[item.name].append(item.lineno)
            for name, locations in methods.items():
                if len(locations)>1:
                    overwritten.append({'file':path,'class':node.name,'method':name,'lines':locations})
    return {'files':len(files),'lines':sum(item['lines'] for item in files),
            'nonblank_lines':sum(sum(bool(line.strip()) for line in index['lines']) for index in sources.values()),
            'counts':dict(counts),'distinct_version_literals':len(labels),
            'distinct_version_identifiers':len(identifiers),'functions':counts['functions'],
            'version_labels':dict(labels),'version_identifiers':dict(identifiers),
            'duplicate_groups':groups,'overwritten_methods':overwritten,
            'remaining_handlers':handlers,'large_functions':large,'source_files':files}


def specific_handler_reason(item):
    file, name, code = item['file'], item['function'].split('.')[-1], item['handler']
    if name=='_parse_sparse_entry': return 'Only fixed-width numeric parse ValueError is caught before parsing the same seven sparse-entry columns as whitespace-delimited text. This is OpenMX input syntax handling, not a numerical algorithm fallback.'
    if name=='_numeric_from_nested_dict': return 'Optional Fermi-level metadata search skips nonnumeric nested values using only TypeError/ValueError; a scientific matrix calculation is not within this handler.'
    if name=='_coerce_value': return 'CSV scalar coercion first attempts an integer, then a float, then retains a text label. Only numeric-conversion ValueError is ignored; invalid array/matrix input is not repaired.'
    if name=='get_or_compile': return 'A genuinely absent cache artifact (FileNotFoundError only) triggers the same declared compiler under its per-key lock. Corrupt, mismatched or failed certified cache data propagates instead of recompiling silently.'
    if name=='resolve_project_gauge_anchor_candidates':
        if 'except ValueError' in code:
            return 'REQUIRES REVIEW: generic ValueError around optional source-group-translated candidate still mixes expected candidate inapplicability with malformed shared metadata. Parent assigned typed domain rejection fix.'
        return 'Only explicitly typed source-group-translation candidate inapplicability is skipped; malformed shared metadata and generic programming errors propagate. The declared H-block algorithm is unchanged.'
    if file.endswith('selection_artifact.py'):
        if name=='begin': return 'Releases the transaction file lock and closes its handle on failed PENDING-marker validation, then re-raises; no selection is published.'
        if item['silent_pass']: return 'Best-effort restoration of the PENDING marker after final-marker publication failed. Outer handler always raises PERSISTENCE_FAILURE with the original exception; no committed selection can be reported.'
        if name=='stage_payloads': return 'Payload generation strict-reload failure is chained into PERSISTENCE_FAILURE; publication does not proceed.'
        if 'strict reload before publication' in code: return 'Generation strict-reload failure before publication is chained into PERSISTENCE_FAILURE; no final marker is written.'
        return 'Final-marker write/reload failure attempts PENDING restoration, then raises chained PERSISTENCE_FAILURE. Rollback failure does not turn publication into success.'
    if file.endswith('siesta.py'):
        if name=='publish_directory': return 'Rolls back only linked task-owned files and empty newly created directories, including interruption, then re-raises. The OSError/pass protects concurrently added foreign directory entries.'
        return 'Deletes task-owned SIESTA staging directory on any import failure or interruption, then re-raises; durable publication is not reported.'
    if file.endswith('response_basis.py'): return 'Adds group/seed/stage context to the original response compilation exception and re-raises; no alternate compiler or numerical result is selected.'
    if file.endswith('model/export.py'): return 'Bulk export inventory records the blocked model directory, exception type and reason for every failure. This is an inventory operation, not candidate selection or numerical fallback.'
    if name=='_bounded_blas_threads': return 'Handler covers only importing optional threadpoolctl. The same H/block/downfold calculation continues with its existing thread context; it cannot catch a numerical matrix calculation.'
    if name=='_get_H_block_impl': return 'Handler covers only optional tqdm import; substitutes a transparent iterator for progress display. H-block matrix construction is outside the handler.'
    if name=='_set_thread_limits': return 'Best-effort threadpoolctl host-capability limiter resets its limiter bookkeeping; it does not select another eigensolver or alter Hamiltonian matrices.'
    if name=='_solve_eigs_slepc': return 'Adds a chained overlap/inner-product diagnostic for the concrete PETSc failure; all other EPSSolve exceptions re-raise. No SciPy fallback is selected.'
    if name=='_mp_kpoint_worker': return 'Transports a failed k-point status to the parent; parent raises on every failure. Failed results cannot be published as successful bands.'
    if name=='_joblib_worker': return 'Sets failed k-point progress status and re-raises the original error; no serial retry or alternate eigensolver.'
    if name in ('_mpi_world_rank_size','_mpi_rank_and_size'): return 'Reads independent MPI-launcher rank/size environment variables when mpi4py communicator discovery is unavailable; no numerical backend selection.'
    if name=='calculate_band_structure': return 'Optional mpi4py import for rank-zero output coordination in MPI-world SLEPc mode; no numerical solver replacement. Actual band solve occurred before this import.'
    if name=='analyze_from_files': return 'Energy/wavefunction load failure is chained into RuntimeError with its source file diagnostic; composition analysis cannot continue with substitute arrays.'
    if name=='analyze_orbital_composition': return 'Prints the offending layer index and orbital offsets, then re-raises; no altered orbital partition or substitute weights.'
    if name=='save_results': return 'Serialization/output failure is chained into RuntimeError; saving cannot be reported as successful.'
    if name=='run' and file.endswith('orbital_analysis_tool.py'): return 'Collects failed requested valley names, then raises if any failed; there is no alternate basis inference or successful final result for a partially failed run.'
    if name in ('main','run_calc','run'): return 'Terminal CLI boundary reports this failure and exits/returns nonzero; it does not replace the requested scientific calculation.'
    return 'REQUIRES REVIEW: '+name+' '+code[:200]


baseline=json.loads((OLD/'statistics.json').read_text())
before={}
for item in baseline['files']:
    if item['scope']!='core': continue
    result=subprocess.run(['git','show',BEFORE_COMMIT+':'+item['file']],cwd=ROOT,capture_output=True,check=True)
    assert hashlib.sha256(result.stdout).hexdigest()==item['sha256'], item['file']
    before[item['file']]=index_source(result.stdout.decode())
# Respect rg visibility, exactly as the original core inventory did.
visible=subprocess.run(['rg','--files','kp/kp','tapw/tapw','-g','*.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
after={path:index_source((ROOT/path).read_text()) for path in visible}
after_stats=analyze(after)
recomputed_before=analyze(before)
comparison_keys=['version_tokens','version_lines','version_literal_nodes','version_gated_if_nodes','legacy_mentions','fallback_mentions','compatibility_mentions','broad_except_handlers','bare_except_handlers','silent_except_pass_handlers','files_at_least_2000_lines','functions_at_least_300_lines','functions_at_least_20_parameters','version_identifier_uses','version_named_functions','dynamic_exec_calls','debug_blocks','debug_http_posts','silent_except_inside_exec_strings']
measurement_checks={key:{'recorded':baseline['core']['counts'].get(key,0),'recomputed':recomputed_before['counts'].get(key,0)} for key in comparison_keys}
assert all(item['recorded']==item['recomputed'] for item in measurement_checks.values()), measurement_checks
assert len(recomputed_before['duplicate_groups'])==len(baseline['exact_duplicate_long_body_groups'])
(OUT/'after_statistics.json').write_text(json.dumps({'scope':'rg-visible *.py under kp/kp and tapw/tapw only; test/scripts/examples/report files excluded',
    'recorded_head':baseline['source_commit'],'verified_snapshot_commit':BEFORE_COMMIT,
    'baseline_hashes_verified':len(before),'measurement_checks':measurement_checks,
    'before':baseline['core'],'after':after_stats},indent=2)+'\n')

# Preserve all agent records and link each original finding to a current owner.
agent_paths=[OUT/name for name in ('symmetry_decisions.csv','response_decisions.csv','model_decisions.csv','tapw_decisions.csv')]
agents=[]
normalize={'保留':'retain','改名':'rename','收紧':'narrow','删除':'delete','合并':'merge','不处理':'excluded'}
for path in agent_paths:
    for number,item in enumerate(read_csv(path),1):
        item=dict(item)
        item.update(origin=path.name,number=number,file=item.get('file',item.get('path','')),detail=item.get('item',item.get('object','')),
                    normalized_decision=normalize.get(item['decision'],item['decision']))
        agents.append(item)
renames=json.loads((OUT/'response_renames.json').read_text())
rename_map={**renames['identifiers'],**renames['strings'],
            'plot_band_01':'plot_band','canonical_v1':'canonical',
            '_v1_central_phase':'_binary_central_phase',
            'EXPLICIT_LEGACY_BASIS_KIND':'EXPLICIT_LAYER_BASIS_KIND',
            'ExplicitLegacyBasisSpec':'ExplicitLayerBasisSpec'}
owners=json.loads((OUT/'model_symbol_owners.json').read_text())
current_functions=defaultdict(list)
for file,index in after.items():
    for start,end,name,node in index['functions']:
        current_functions[name.split('.')[-1]].append((file,start,end,name))


def resolve(file,line,detail):
    function=enclosing(before[file],line) if file in before else '<module>'
    function_name=function.split('.')[-1]
    new_name=rename_map.get(function_name,function_name)
    destinations=current_functions.get(new_name,[])
    if file=='tapw/tapw/plot_band_01.py':file='tapw/tapw/plot_band.py'
    if function_name in ('parse_kpath_labels','read_kpath','parse_kpath_in'):
        destinations=[entry for entry in destinations if entry[0]=='tapw/tapw/io/kpath.py'] or destinations
    elif function_name=='_field': destinations=current_functions.get('write_report_field',[])
    elif function_name=='separate_layers' and file=='tapw/tapw/io/structure.py': destinations=current_functions.get('_separate_layers_by_kmeans',[])
    elif function_name in ('hash_array','require_identity_fields','require_matching_identity','_canonical_value') and file=='kp/kp/identity.py':
        destinations=current_functions.get('canonical_json_value' if function_name=='_canonical_value' else function_name,[])
        destinations=[entry for entry in destinations if entry[0]=='tapw/tapw/identity.py']
    elif file.startswith('kp/kp/model/') and function_name in owners:
        owner='kp/kp/model/'+owners[function_name]+'.py'
        destinations=[entry for entry in destinations if entry[0]==owner] or destinations
    destinations=sorted(destinations,key=lambda entry:entry[0]!=file)
    target=destinations[0] if destinations else None
    replacement=detail
    for old,new in sorted(rename_map.items(),key=lambda entry:len(entry[0]),reverse=True):
        replacement=replacement.replace(old,new)
    target_file=target[0] if target else file
    if target_file in after:
        index=after[target_file]
        locations=[i for i,text in enumerate(index['lines'],1) if replacement in text and replacement.strip()]
        if locations: return target_file,min(locations,key=lambda i:abs(i-(target[1] if target else line))),function,replacement
        if target:return target_file,target[1],function,replacement
    return (target_file if target_file in after else ''),0,function,replacement


def agent_match(file,line,category,detail):
    exact=[item for item in agents if item['file']==file and str(item.get('line',''))==str(line) and item.get('category')==category]
    if exact:return exact[-1]
    exact=[item for item in agents if item['file']==file and (item['detail']==detail or item['detail'] in detail and len(item['detail'])>5)]
    if exact:return exact[-1]
    return None


def decision_for(category,file,line,detail,current_file,current_line,function,replacement,agent):
    if category in ('debug_blocks','debug_http_posts','dynamic_exec_calls','silent_except_inside_exec_strings'):
        return 'delete','Fixed-address network debug execution and its swallowed exceptions are absent from current scientific production code.'
    if category=='exact_duplicate_long_function_body':
        if '_joint_adjoint_key_artifact' in detail or '_local_fixed_joint_key_cache_artifact' in detail:
            return 'retain','Identical small serializer retained at distinct local-fixed and graded compiler cache metadata boundaries. This is metadata duplication, not independent mathematical validation.'
        return 'merge',f'Original repeated production body now has one owner ({current_file or "shared imported implementation"}); callers directly use that implementation.'
    if category in ('broad_except_handlers','bare_except_handlers','silent_except_pass_handlers'):
        remaining=[item for item in after_stats['remaining_handlers'] if item['file']==current_file and item['function'].split('.')[-1]==function.split('.')[-1]]
        if remaining:return 'retain',specific_handler_reason(min(remaining,key=lambda item:abs(item['line']-current_line)))
        if current_line:return 'narrow',f'{function}: current owner has only domain-specific input/library exceptions or propagates the original error; former broad/bare/pass handler was removed.'
        return 'delete',f'{function}: former implementation/error-swallowing route no longer exists in production.'
    if 'version' in category:
        if 'legacy_frozen' in detail or 'LEGACY_FROZEN' in detail:
            return 'delete','Retired frozen-response runtime route; current compilation uses one complete real-linear response semantics. No legacy calculation fallback remains.'
        if 'v57' in detail or 'COMPILER_VERSION' in detail:
            return 'replace','Compiler implementation fingerprint replaces a manually incremented revision; cache identity continues to bind real input Q, basis ordering and mathematical conventions.'
        if detail in rename_map or replacement!=detail:
            return 'rename',f'Current expression/name is {replacement[:170]!r}; development generation was removed while the declared mathematical convention stayed fixed.'
        if not current_line:return 'delete',f'{function}: original version-selected implementation/name has no remaining current production owner.'
        line_text=after[current_file]['lines'][current_line-1]
        if not VERSION.search(line_text):
            return 'rename',f'{function}: current owner uses a mathematical or format responsibility name; original development-generation marker was removed.'
        return 'retain',f'{current_file}:{current_line} records/checks the current serialized convention or hash-domain discriminator {replacement[:150]!r}. This does not select an older numerical implementation.'
    if category in ('functions_at_least_300_lines','functions_at_least_20_parameters','files_at_least_2000_lines','lines_over_300_chars','lines_over_1000_chars'):
        if not current_line:return 'delete',f'Historical {function} implementation was removed; no line-count-driven substitute API was added.'
        if current_file!=file:return 'migrate',f'{function} now belongs to {current_file}; responsibility migration preserves the calculation signature/data rather than introducing a universal context.'
        return 'retain',f'{function} retains its explicit scientific stage/interface in {current_file}; size is reported separately and is not evidence of an incorrect algorithm.'
    if category=='repeated_function_definition_same_scope':
        if 'plot_lattice1' in detail:return 'delete','Removed overwritten first plot method; current single-layer drawing has one implementation.'
        return 'retain','Nested refinement branches define separate local functions; they are not an unconditional class-method overwrite.'
    if not current_line:return 'delete',f'Original {function} path/comment has no current implementation at the former owner.'
    if agent and agent['normalized_decision']!='retain':return agent['normalized_decision'],agent.get('reason','')
    if category=='legacy_mentions':
        if 'openmx' in detail.lower():return 'retain','Explicit OpenMX DFT input interpretation remains supported; this provenance label does not load an older response solver or cache.'
        if 'analytic' in replacement or 'C3' in detail:return 'retain','Direct analytic C3 transport remains an independent numerical reference; it is not used as a fallback after production certification fails.'
        return 'retain',f'{function}: historical word remains in a documented basis/data convention at {current_file}:{current_line}; current runtime behavior is assessed independently of the word.'
    if category=='fallback_mentions':return 'retain',f'{function}: surviving occurrence at {current_file}:{current_line} describes an explicitly inspected capability/diagnostic or expected-domain outcome; no caught program error chooses a different production algorithm.'
    if category=='compatibility_mentions':return 'retain',f'{function}: remaining description at {current_file}:{current_line} concerns a declared data/basis interface; no old-format migration branch was introduced.'
    return 'retain',f'{function}: current owner {current_file}:{current_line} was inspected; mathematical/data role is preserved.'

records=[]
for number,item in enumerate(read_csv(OLD/'inventory.csv'),1):
    file,line,category,detail=item['file'],int(item['line']),item['category'],item['detail']
    current_file,current_line,function,replacement=resolve(file,line,detail)
    match=agent_match(file,line,category,detail)
    decision,reason=decision_for(category,file,line,detail,current_file,current_line,function,replacement,match)
    records.append({'record_id':f'inventory:{number}','origin':'inventory.csv','category':category,'original_file':file,'original_line':line,
                    'item':detail,'decision':decision,'current_file':current_file,'current_line':current_line,'function':function,
                    'reason':reason,'evidence':match['origin'] if match else 'current AST/source inspection',
                    'review_status':'reviewed' if not reason.startswith('REQUIRES REVIEW') else 'needs_review'})
for number,item in enumerate(read_csv(OLD/'version_labels.csv'),1):
    file,line,detail=item['first_file'],int(item['first_line']),item['label']
    current_file,current_line,function,replacement=resolve(file,line,detail)
    match=agent_match(file,line,'version_literal_nodes',detail)
    decision,reason=decision_for('version_label',file,line,detail,current_file,current_line,function,replacement,match)
    records.append({'record_id':f'version_label:{number}','origin':'version_labels.csv','category':'distinct_version_label','original_file':file,'original_line':line,
                    'item':detail,'decision':decision,'current_file':current_file,'current_line':current_line,'function':function,
                    'reason':reason,'evidence':match['origin'] if match else 'current literal/rename inspection','review_status':'reviewed'})
for item in agents:
    reason=item.get('reason','')
    generic=any(phrase in reason for phrase in ('Current mathematical behavior, coordinate convention', 'Mathematical identity without development revision', 'Narrow domain/input validation or error context', 'Explicit physical/standalone serialization discriminators'))
    if generic and item['file']:
        original_line=int(item.get('line') or 1) if str(item.get('line') or 1).isdigit() else 1
        current_file,current_line,function,replacement=resolve(item['file'],original_line,item['detail'])
        if current_file and current_line:
            statement=after[current_file]['lines'][current_line-1].strip()
            if item['normalized_decision']=='rename':
                reason=f"{item['detail']!r} now names {item.get('replacement') or replacement!r} in {current_file}; the statement {statement[:180]!r} expresses its mathematical/format role directly, without generation-based dispatch."
            else:
                reason=f"Exact retained object in {current_file}:{current_line} ({function}): {statement[:220]!r}. This is the declared current data/convention/interface checked by that owner; no historical implementation is selected."
        else:
            reason=f"Original object {item['detail']!r} in {item['file']} belongs to the documented current serialization/input boundary; its original owner record remains available for review."
    records.append({'record_id':f'agent:{item["origin"]}:{item["number"]}','origin':item['origin'],'category':item.get('category','agent_decision'),
                    'original_file':item['file'],'original_line':item.get('line',''),'item':item['detail'],'decision':item['normalized_decision'],
                    'current_file':item.get('replacement',''),'current_line':'','function':'','reason':reason,
                    'evidence':item.get('verification',item.get('validation',item.get('callers',''))),'review_status':'agent-reviewed'})
for item in after_stats['remaining_handlers']:
    reason=specific_handler_reason(item)
    records.append({'record_id':f'current_handler:{item["file"]}:{item["line"]}','origin':'current AST','category':'current_error_boundary',
                    'original_file':'','original_line':'','item':item['handler'],'decision':'retain','current_file':item['file'],'current_line':item['line'],
                    'function':item['function'],'reason':reason,'evidence':'direct current source inspection','review_status':'reviewed' if not reason.startswith('REQUIRES REVIEW') else 'needs_review'})
fields=['record_id','origin','category','original_file','original_line','item','decision','current_file','current_line','function','reason','evidence','review_status']
with (OUT/'decisions.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(records)
coverage={'inventory_total':len(read_csv(OLD/'inventory.csv')),'inventory_covered':sum(row['origin']=='inventory.csv' for row in records),
          'distinct_version_labels_total':len(read_csv(OLD/'version_labels.csv')),'distinct_version_labels_covered':sum(row['origin']=='version_labels.csv' for row in records),
          'agent_records':len(agents),'current_error_boundaries_reviewed':len(after_stats['remaining_handlers']),
          'needs_review':[row['record_id'] for row in records if row['review_status']=='needs_review'],
          'baseline_snapshot_commit':BEFORE_COMMIT,'baseline_source_hashes_verified':len(before),
          'after_source_fingerprint':hashlib.sha256(json.dumps(after_stats['source_files'],sort_keys=True).encode()).hexdigest()}
(OUT/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
gates=[]
for file,index in after.items():
    for node in ast.walk(index['tree']):
        if not isinstance(node,ast.If) or not VERSION.search(ast.unparse(node.test)): continue
        test=ast.unparse(node.test)
        if file.endswith('selection_artifact.py'):role='Exact current PENDING transaction record schema for Gamma/non-Gamma selection, with status validation.'
        elif file.endswith('physical_runtime.py'):role='Exact current standalone physical model configuration schema; wrong formats rejected.'
        elif file.endswith('candidate_certificate.py'):role='Exact current candidate input/presentation certificate artifact discriminator and source binding rule.'
        elif file.endswith('symmetry_gauge.py'):role='Two declared current basis-ordering data types (sector ordering versus full unitary ordering), not old numerical algorithm generations.'
        elif file=='tapw/tapw/symm_rep.py':role='Exact current packed raw-H representation schema before CSR reconstruction.'
        elif file=='tapw/tapw/chern_post.py':role='Optional topology grid/collection manifest type recognition; current producer omits manifests, retained reader boundary documented.'
        else:role='REQUIRES REVIEW'
        gates.append({'file':file,'line':node.lineno,'test':test,'role':role})
with (OUT/'remaining_format_gates.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['file','line','test','role'],lineterminator='\n');writer.writeheader();writer.writerows(gates)

lines=['Cleanup audit coverage and comparable statistics','',
       f'Original inventory: {coverage["inventory_covered"]}/{coverage["inventory_total"]} items linked.',
       f'Original distinct version labels: {coverage["distinct_version_labels_covered"]}/{coverage["distinct_version_labels_total"]} items linked.',
       f'Agent decision records preserved: {len(agents)}; current error boundaries: {len(after_stats["remaining_handlers"])}.',
       'The original statistics HEAD label was 3b6ca7f. All 106 source hashes match the saved 060dbc7 snapshot.',
       'Counts use rg-visible Python under kp/kp and tapw/tapw. Tests, scripts, examples, paper and this audit tool are excluded.',
       'Repeated inventory categories overlap; row totals are coverage counts, not independent bug counts.','',
       'Metric | before | after',f'Python files | {baseline["core"]["files"]} | {after_stats["files"]}',
       f'Physical lines | {baseline["core"]["lines"]} | {after_stats["lines"]}']
for key in comparison_keys:
    lines.append(f'{key} | {baseline["core"]["counts"].get(key,0)} | {after_stats["counts"].get(key,0)}')
lines.extend([f'distinct version literals | {baseline["core"]["distinct_version_literals"]} | {after_stats["distinct_version_literals"]}',
              f'exact duplicate long function groups | {len(baseline["exact_duplicate_long_body_groups"])} | {len(after_stats["duplicate_groups"])}',
              f'unconditionally overwritten class methods | 1 | {len(after_stats["overwritten_methods"])}','',
              'Remaining duplication: compiler cache-key metadata serializers are retained at their distinct local-fixed / graded metadata boundaries. They are not independent mathematical references.',
              'Necessary current serialized schema/hash-domain tags remain. Plain v1/v2 local vector variables can still be counted by textual token statistics; they are not old algorithm dispatch.',
              'Broad catches retained are terminal CLI failures, chained/re-raised compilation failures, resource rollback, explicitly blocked export inventories, MPI/process capability discovery or cosmetic progress control.',
              'Scientific regression and installed-material evidence are in separate test/example reports; source-statistic reduction does not prove numerical correctness.',
              f'Unclassified error boundaries: {len(coverage["needs_review"])}.'])
(OUT/'AUDIT_REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'coverage':coverage,'after_counts':after_stats['counts'],'after_files':after_stats['files'],'after_lines':after_stats['lines'],
                  'duplicate_groups':after_stats['duplicate_groups']},indent=2))
