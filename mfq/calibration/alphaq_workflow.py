"""Normal HF quantization from an imatrix and a complete-file BPW ceiling."""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, replace
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from mfq.architectures.hf_config import text_section
from mfq.calibration.alphaq import ALPHAQ_PROFILES, alphaq_builtin_candidates
from mfq.calibration.alphaq_cache import collect_statistics, source_identity, trained_tables
from mfq.calibration.alphaq_joint import allocate_joint, dense_choices
from mfq.calibration.alphaq_model import full_file_budget, imatrix_router_multipliers, model_scope
from mfq.calibration.alphaq_output import AlphaQOutput, file_sha256, model_files, runtime_metadata
from mfq.calibration.alphaq_source import _atomic_json, _progress
from mfq.calibration.artifact import CalibrationScheme, save_scheme, load_scheme
from mfq.quantize.imatrix import load_importance_matrix
from mfq.calibration.ud_recipes import automatic_recipe, registered_model


def recipe_types(path, *, text_layers=None):
    from mfq.tools import quantize_hf_to_mfq as q
    # Experts have their own complete candidate set, including formats absent
    # from the source GGUF. Ignore their recipe types before the strict loader.
    if path.suffix.lower() == '.json':
        doc = json.loads(path.read_text())
        if doc.get('format') != 'mfq.gguf-recipe.v1' or not isinstance(doc.get('tensor_types'), dict):
            raise ValueError('unsupported AlphaQ GGUF recipe document')
        values = doc['tensor_types']
    else:
        values = q._load_gguf_recipe(path)
    return {name: ('Q8_0' if name.endswith('_exps.weight') and
                   (text_layers is None or int(name.split('.')[1]) < text_layers) else dtype)
            for name, dtype in values.items()}


def planned(root, recipe, scheme, overrides, ple, *, vision='exclude', mtp='exclude',
            inventory=None, config=None):
    from mfq.tools import quantize_hf_to_mfq as q
    plans = q._plan(root, vision == 'exclude', recipe, 'F32', scheme,
        quantize_ple=True, quantize_mtp=mtp == 'quantize', exclude_mtp=mtp == 'exclude',
        source_inventory=inventory, source_config=config)
    if vision == 'NINT8':
        overrides = {**{p.name: 'NINT8' for p in plans
                        if p.name.startswith('vision.') and len(p.shape) == 2}, **overrides}
    return q._normalize_hf_expert_storage(q._apply_tensor_precision_overrides(plans, overrides))


def routing(scope, imatrix, config, mode, cache_path, root, *, absolute=False):
    from mfq.tools import quantize_hf_to_mfq as q
    topk = text_section(config).get('num_experts_per_tok')
    if type(topk) is not int or topk <= 0:
        raise ValueError('AlphaQ needs num_experts_per_tok in the model configuration')
    if mode == 'none':
        if cache_path:
            raise ValueError('--alphaq-router-cache requires router frequency weighting')
        return None, {p.name: topk/p.shape[0] for p in scope.experts}
    if imatrix.metadata.get('objective') == 'uniform_calibration_free' and not cache_path:
        raise ValueError('uniform imatrix has no observed router counts; use --alphaq-router none '
                         'or provide --alphaq-router-cache with measured routing frequencies')
    router = {}
    for item in scope.plans:
        match = re.fullmatch(r'model\.block\.(\d+)\.mlp\.router\.weight', item.name)
        if match:
            found = imatrix.find(q._hf_imatrix_names(item))
            if found:
                router[int(match[1])] = found[0]
    if cache_path:
        doc = json.loads(Path(cache_path).read_text())
        if (doc.get('format') != 'mfq.alphaq-router-frequency.v1'
                or doc.get('source') != source_identity(root) or doc.get('topk') != topk):
            raise ValueError('router frequency cache does not match this source model/topk')
        layers = {int(k): v for k, v in doc['layers'].items()}
        expected = {int(p.name.split('.')[2]) for p in scope.experts}
        if set(layers) != expected or set(router) != expected:
            raise ValueError('router cache must cover the complete model')
        entries = dict(imatrix.entries)
        for layer, name in router.items():
            entries[name] = replace(entries[name], counts=np.asarray([layers[layer]['tokens']]))
        for item in scope.experts:
            layer = int(item.name.split('.')[2]); name = scope.imatrix_entries[item.name]
            entries[name] = replace(entries[name], counts=np.asarray(layers[layer]['counts']))
        imatrix = replace(imatrix, entries=entries)
    return imatrix_router_multipliers(scope, imatrix, router, topk, normalize=not absolute)


def allocation_signature(key, cache, budget):
    directory = Path(__file__).parent
    package = directory.parent
    paths = set(directory.glob('*.py')) | set((package/'formats').glob('*.py'))
    paths.update(package/name for name in ('tools/quantize_hf_to_mfq.py',
        'quantize/standard_presets.py', 'quantize/imatrix.py'))
    statistics = json.loads((cache/'statistics.json').read_text())
    statistics_digest = hashlib.sha256(json.dumps(statistics, sort_keys=True).encode()).hexdigest()
    return dict(key=key, statistics_sha256=statistics_digest,
                budget=asdict(budget), implementation={str(path.relative_to(package)):file_sha256(path)
                    for path in sorted(paths)})


def saved_allocation(directory, signature):
    marker = directory/'allocation-checkpoint.json'
    if not marker.exists():
        return None
    document = json.loads(marker.read_text())
    if document.get('signature') != signature:
        return None
    names = {'untrained-scheme.json', 'allocation.json', 'native-overrides.json'}
    if document.get('complete') is not True or set(document.get('files', {})) != names:
        raise ValueError('invalid AlphaQ allocation checkpoint')
    for name, digest in document['files'].items():
        if file_sha256(directory/name) != digest:
            raise ValueError('AlphaQ allocation checkpoint checksum differs')
    return (load_scheme(directory/'untrained-scheme.json'),
            json.loads((directory/'native-overrides.json').read_text()))


def quantize(args):
    from mfq.calibration.alphaq_cache import cache_lock
    output = Path(args.output).resolve()
    cache = Path(args.alphaq_cache).resolve() if args.alphaq_cache else output.parent/(output.name+'.alphaq')
    with cache_lock(cache/'.quantize.lock'), ExitStack() as resources:
        return _quantize(args, resources)


def _quantize(args, resources):
    from mfq.commands.quantize import _hf_arguments
    from mfq.tools import quantize_hf_to_mfq as q
    from mfq.quantize.backend import resolve_quant_backend

    root, output = Path(args.input).resolve(), Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    cache = Path(args.alphaq_cache).resolve() if args.alphaq_cache else output.parent/(output.name+'.alphaq')
    cache.mkdir(parents=True, exist_ok=True)
    config = q.load_hf_model_config(root)
    recipe_path = (Path(args.recipe).resolve() if args.recipe else
                   automatic_recipe(config, args.target_bpw, tier=args.ud_tier or None))
    ceiling = None
    if args.ud_tier:
        doc = json.loads(recipe_path.read_text())
        model = registered_model(config)
        if model is None or args.ud_tier not in model['caps']:
            raise ValueError('--ud-tier requires a registered model configuration and tier')
        ceiling = doc.get('ud_file_bytes', model['caps'][args.ud_tier])
    target_bpw = args.target_bpw or '1'  # Exact UD byte ceiling overrides BPW below.
    mode = args.alphaq_dense
    if mode == 'auto':
        mode = 'recipe' if recipe_path is not None else 'joint'
    if mode == 'recipe' and recipe_path is None:
        raise ValueError('recipe dense mode requires --recipe for this model')
    if args.alphaq_allocator == 'natscale':
        if mode != 'recipe':
            raise ValueError('NatScale requires recipe dense mode; joint whole-model units are not implemented')
        if args.alphaq_router != 'imatrix' or args.alphaq_router_cache:
            raise ValueError('NatScale uses validated imatrix exposure; router overrides are not supported')
        if not np.isfinite(args.natscale_tau) or args.natscale_tau <= 0:
            raise ValueError('NatScale tau must be finite and positive')
        if args.natscale_sample_layers < 1 or args.natscale_sample_experts < 1:
            raise ValueError('NatScale sample sizes must be positive integers')
        if args.natscale_sensitivity not in ('uniform', 'imatrix-energy') and not args.natscale_sensitivity.startswith('gradient:'):
            raise ValueError('NatScale sensitivity must be uniform, imatrix-energy or gradient:PATH')
    recipe = recipe_types(recipe_path) if recipe_path is not None else None
    imatrix = load_importance_matrix(Path(args.imatrix).resolve())
    imatrix_sha = file_sha256(imatrix.path)
    empty = CalibrationScheme(None, 'AlphaQ', 0, {}, {}, {})
    inventory = q._hf_source_inventory(root)
    options = dict(vision='exclude' if args.text_only else args.alphaq_vision,
                   mtp='exclude' if args.exclude_mtp else args.alphaq_mtp,
                   inventory=inventory, config=config)
    def plan(scheme, overrides, types=recipe):
        return planned(root, types, scheme, overrides, args.ple_dtype, **options)
    plans = plan(empty, {})
    if recipe is not None:
        complete = plan(empty, {}, None)
        if {p.name: p.shape for p in plans} != {p.name: p.shape for p in complete}:
            raise ValueError('recipe changes the complete text model tensor scope')
    scope = model_scope(plans, imatrix, exclude_scaled_dense=mode == 'joint')
    overrides = {}
    if options['vision'] == 'NINT8':
        overrides.update({p.name: 'NINT8' for p in plans if p.name.startswith('vision.') and len(p.shape) == 2})
    # The selected recipe fallback intentionally preserves the complete UD
    # nonexpert policy. Joint allocation preserves missing-imatrix matrices.
    if mode == 'joint':
        overrides.update({p.name: 'BF16' if p.source_quantization else p.source_dtype for p in scope.native_dense})
    for item in scope.ple:
        if len(item.shape) != 2:
            continue
        if args.ple_dtype == 'native' or (args.ple_dtype == 'recipe' and recipe is None):
            overrides[item.name] = item.source_dtype
        elif args.ple_dtype != 'recipe':
            # Lookup matrices are independently configurable; key/value and
            # norms retain the chosen recipe's precision.
            if 'ngram' in item.name:
                overrides[item.name] = args.ple_dtype
    plans = plan(empty, overrides)
    scope = model_scope(plans, imatrix, exclude_scaled_dense=mode == 'joint')
    from mfq.formats.nint import NintSpec
    auxiliary = {component: sum(q._plan_blob_nbytes(p, NintSpec()) for p in plans
                     if p.name.startswith(component+'.')) for component in ('vision', 'predictor')}
    profiles = tuple(args.alphaq_profiles.split(',')) if args.alphaq_profiles else ALPHAQ_PROFILES
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles)-set(ALPHAQ_PROFILES):
        raise ValueError('AlphaQ profiles must be distinct canonical NVQ/NINT profiles')
    policy = dict(method='AlphaQ', distortion='analytical', dense_mode=mode,
        router='native_cache_frequency' if args.alphaq_router_cache else args.alphaq_router,
        router_cache_sha256=file_sha256(Path(args.alphaq_router_cache)) if args.alphaq_router_cache else None,
        recipe_sha256=file_sha256(recipe_path) if recipe_path else None,
        recipe=recipe_path.stem if recipe_path else None, target_bpw=str(args.target_bpw),
        ud_tier=args.ud_tier or None, vision=options['vision'], mtp=options['mtp'],
        vision_bytes=auxiliary['vision'], mtp_bytes=auxiliary['predictor'],
        mtp_imatrix='unweighted: predictor entries are absent from the supplied imatrix',
        joint_scaled_dense='excluded from allocation; decoded BF16 fixed payload',
        unweighted_tensors=[p.name for p in plans if not imatrix.find(q._hf_imatrix_names(p))],
        model_weight_count=scope.model_weight_count, profiles=list(profiles), ple=args.ple_dtype,
        scope='text main model, excludes MTP; integer constants count as bytes only',
        quality_status='allocation surrogate; this output requires full-model evaluation')
    if (args.alphaq_allocator == 'alphaq' and not args.ud_tier and not any(auxiliary.values())
            and not any(p.source_quantization for p in scope.native_dense)):
        # Preserve the existing AlphaQ text-only header and budget arithmetic
        # when none of the new policies apply (including Qwen target-BPW runs).
        for name in ('ud_tier', 'vision', 'mtp', 'vision_bytes', 'mtp_bytes',
                     'mtp_imatrix', 'joint_scaled_dense', 'unweighted_tensors'):
            policy.pop(name)
    if args.alphaq_allocator == 'natscale':
        error_path = Path(args.natscale_error_cache) if args.natscale_error_cache else cache/'natscale-errors.json'
        policy.update(method='NatScale', distortion='separate-error surrogate',
            allocator='natscale', natscale=dict(tau=args.natscale_tau,
            sensitivity=args.natscale_sensitivity, error=args.natscale_error,
            sample_layers=args.natscale_sample_layers, sample_experts=args.natscale_sample_experts,
            error_cache_sha256=file_sha256(error_path)
                if args.natscale_error == 'measured' and error_path.exists() else None,
            gradient_sha256=file_sha256(Path(args.natscale_sensitivity[9:]))
                if args.natscale_sensitivity.startswith('gradient:') else None))
    if not args.recipe and recipe_path is not None and scope.model_weight_count != registered_model(config)['parameters']:
        raise ValueError('automatic UD recipe parameter count differs; specify an explicit recipe')
    allocation_key = hashlib.sha256(json.dumps(dict(policy=policy, imatrix=imatrix_sha,
        source=source_identity(root)), sort_keys=True).encode()).hexdigest()
    allocation_root = cache/'allocations'/allocation_key
    allocation_root.mkdir(parents=True, exist_ok=True)
    if recipe is not None:
        normalized_recipe = allocation_root/'recipe.json'
        _atomic_json(normalized_recipe, {'format': 'mfq.gguf-recipe.v1', 'tensor_types': recipe})
    header, assets = runtime_metadata(root, plans, imatrix=imatrix, imatrix_sha256=imatrix_sha,
        policy=policy, tokenizer=Path(args.tokenizer).resolve() if args.tokenizer else None,
        sampling_profile=args.sampling_profile or None)
    auxiliary_bytes = sum(auxiliary.values())
    budget = full_file_budget(scope, header, assets, target_bpw=target_bpw,
        maximum_file_bytes=ceiling, excluded_payload_bytes=auxiliary_bytes,
        joint_dense=mode == 'joint', preserve_missing_imatrix=mode == 'joint')
    _progress(stage='alphaq_file_budget', maximum_file_bytes=budget.maximum_file_bytes,
        fixed_file_bytes=budget.fixed_file_bytes, parameters=scope.model_weight_count,
        dense_mode=mode, expert_banks=len(scope.experts), dense_matrices=len(scope.dense),
        variable_expert_budget=budget.maximum_file_bytes-budget.fixed_file_bytes,
        vision_bytes=auxiliary['vision'], mtp_bytes=auxiliary['predictor'])
    if args.dry_run:
        return
    backend = resolve_quant_backend(args.backend, args.device)
    from mfq.calibration.alphaq_parallel import execution_devices, DeviceWorkers, train_and_fit
    devices = execution_devices(getattr(args, 'devices', ''), backend, args.device)
    workers = resources.enter_context(DeviceWorkers(root, cache, imatrix.path, imatrix_sha,
        backend, devices, row_chunk=args.row_chunk, artifact_root=allocation_root)) if devices else None
    multipliers, exposure = routing(scope, imatrix, config, args.alphaq_router, args.alphaq_router_cache, root,
                                    absolute=args.alphaq_allocator == 'natscale')
    errors = None
    if args.alphaq_allocator == 'natscale' and args.natscale_error == 'measured':
        from mfq.calibration.natscale_errors import load_errors
        error_path = Path(args.natscale_error_cache) if args.natscale_error_cache else cache/'natscale-errors.json'
        errors = load_errors(error_path, root, scope.experts, imatrix_sha, profiles,
                             args.natscale_sample_layers, args.natscale_sample_experts)
    allocated_dense = scope.dense if mode == 'joint' else ()
    statistics = collect_statistics(root, (*scope.experts, *allocated_dense), cache, backend.device,
                                    mapper=workers.map if workers else None,
                                    column_energy=args.alphaq_allocator == 'natscale')
    experts, dense = statistics[:len(scope.experts)], statistics[len(scope.experts):]
    signature = allocation_signature(allocation_key, cache, budget)
    previous = saved_allocation(allocation_root, signature)
    if previous is None:
        if args.alphaq_allocator == 'natscale':
            from mfq.calibration.natscale import allocate_natscale, projection_energies
            frequencies = multipliers
            allocated = allocate_natscale(experts, alphaq_builtin_candidates(experts, profiles),
                energies=projection_energies(experts, scope, imatrix), exposures=frequencies,
                model_weight_count=scope.model_weight_count, maximum_file_bytes=budget.maximum_file_bytes,
                fixed_file_bytes=budget.fixed_file_bytes, tau=args.natscale_tau,
                sensitivity=args.natscale_sensitivity, errors=errors, source=source_identity(root))
        else:
            allocated = allocate_joint(experts, alphaq_builtin_candidates(experts, profiles),
                dense, dense_choices(dense, {p.name: p.source_dtype for p in allocated_dense}, profiles),
                model_weight_count=scope.model_weight_count, maximum_file_bytes=budget.maximum_file_bytes,
                fixed_file_bytes=budget.fixed_file_bytes, expert_exposure=exposure, router_multipliers=multipliers)
        overrides.update(budget.native_overrides); overrides.update(allocated.native_overrides)
        raw_path = allocation_root/'untrained-scheme.json'
        if raw_path.exists():
            if replace(load_scheme(raw_path), path=None) != replace(allocated.scheme, path=None):
                raise ValueError('saved raw AlphaQ allocation differs from reproducible allocation')
        else:
            save_scheme(raw_path, allocated.scheme)
        _atomic_json(allocation_root/'allocation.json', {**allocated.report, 'policy': policy})
        if args.alphaq_allocator == 'natscale':
            _progress(stage='natscale_allocation', **allocated.report)
        _atomic_json(allocation_root/'native-overrides.json', overrides)
        _atomic_json(allocation_root/'allocation-checkpoint.json', dict(complete=True,
            signature=signature, files={name:file_sha256(allocation_root/name) for name in
                ('untrained-scheme.json', 'allocation.json', 'native-overrides.json')}))
        untrained = allocated.scheme
    else:
        untrained, overrides = previous
        _progress(stage='alphaq_allocation_reused', directory=str(allocation_root))
    encoded = None
    if workers is not None and not (cache/'encoded-candidates.json').exists():
        pending_plans = plan(untrained, overrides)
        scheme, encoded = train_and_fit(workers, root, pending_plans, untrained, imatrix,
            imatrix_sha, cache, backend, row_chunk=args.row_chunk, artifact_root=allocation_root)
    else:
        scheme = trained_tables(root, plans, untrained, imatrix, imatrix_sha, cache, backend.device)
    scheme_path = allocation_root/'scheme.json'
    if scheme_path.exists():
        if replace(load_scheme(scheme_path), path=None) != replace(scheme, path=None):
            raise ValueError('saved AlphaQ allocation differs from reproducible allocation')
    else:
        save_scheme(scheme_path, scheme)
    override_path = allocation_root/'native-overrides.json'
    _atomic_json(override_path, overrides)
    low = _hf_arguments(args, output)
    low.recipe_gguf = str(normalized_recipe) if recipe is not None else ''
    low.calibration_scheme = str(scheme_path)
    low.tensor_precision_overrides = str(override_path) if overrides else ''
    low.text_only = options['vision'] == 'exclude'; low.exclude_mtp = options['mtp'] == 'exclude'
    low.quantize_mtp = options['mtp'] == 'quantize'; low.quantize_ple = True; low.dense_dtype = 'f32'
    final_plans = plan(scheme, overrides)
    if (cache/'encoded-candidates.json').exists():
        from mfq.calibration.alphaq_payloads import PayloadCache
        encoded = PayloadCache(cache/'encoded-candidates.json', root, imatrix_sha)
    elif encoded is None:
        from mfq.calibration.alphaq_fit import FittedPayloadCache
        encoded = FittedPayloadCache(root, cache, imatrix, imatrix_sha, backend,
            row_chunk=args.row_chunk, artifact_root=scheme_path.parent)
    low._alphaq_output = AlphaQOutput(root, tuple(final_plans), model_files(root), header, assets,
                                     budget.maximum_file_bytes, encoded, ('vision', 'predictor'))
    q.convert(low)
    from mfq.formats.io import open_mmap
    with open_mmap(output) as store:
        actual_auxiliary = {c: sum(r.nbytes for r in store.records.values()
                                   if r.name.startswith(c+'.')) for c in ('vision', 'predictor')}
    export = dict(output=str(output), file_bytes=output.stat().st_size,
        maximum_file_bytes=budget.maximum_file_bytes, vision_bytes=actual_auxiliary['vision'],
        mtp_bytes=actual_auxiliary['predictor'],
        main_text_file_bytes=output.stat().st_size-sum(actual_auxiliary.values()),
        model_weight_count=scope.model_weight_count)
    _atomic_json(allocation_root/'export-report.json', export)
    _progress(stage='alphaq_complete', file_bytes=output.stat().st_size,
        maximum_file_bytes=budget.maximum_file_bytes,
        vision_bytes=actual_auxiliary['vision'], mtp_bytes=actual_auxiliary['predictor'],
        actual_bpw=export['main_text_file_bytes']*8/scope.model_weight_count, output=str(output))
