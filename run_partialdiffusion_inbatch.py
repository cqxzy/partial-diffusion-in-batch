#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner for RFdiffusion partial diffusion on binder-target complexes.

Key behaviors:
- Input: a directory of complex PDB files (.pdb only).
- Target chain is fixed by --target-chain.
- Binder chain can be provided or auto-inferred as the longest non-target protein chain.
- Builds strict partial-diffusion contig:
  contigmap.contigs=[{binder_len}-{binder_len}/0 {target_chain}{start}-{end}]
- Supports optional residue renumbering to avoid offset issues:
  --renumber-target-to-1 or --renumber-all-to-1
- Executes RFdiffusion via docker image:
  rosettacommons/rfdiffusion:latest (default)
- Logs all details to output_dir/run.log and output_dir/run_summary.tsv
- Supports dry-run and optional flattening of generated PDB files.

Example:
python run_partialdiffusion_inbatch/run_partialdiffusion_inbatch.py \
  --input-dir /data/rfdiffusion-run/xing-work/input/partialdiffusion/test \
  --output-dir /data/rfdiffusion-run/xing-work/output/0215/partialdiffusion \
  --target-chain B \
  --num-designs 20 \
  --partial-T 20 \
  --hotspots 94,127,139,173

Acceptance checks:
1) If target B is 101-314 and binder_len=100:
   contigmap.contigs=[100-100/0 B101-314]
2) With --renumber-all-to-1 and target_len=214:
   contigmap.contigs=[100-100/0 B1-214]
3) Default output layout:
   output_dir/<pdb_stem>/...
4) With --flatten, copy generated .pdb files into:
   output_dir/ALL_PDB/
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gemmi


CANONICAL_AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
}

NUMERIC_HOTSPOT_RE = re.compile(r"^(-?\d+)([A-Za-z]?)$")
CHAINED_HOTSPOT_RE = re.compile(r"^([A-Za-z0-9])(-?\d+)([A-Za-z]?)$")


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resseq: int
    icode: str


@dataclass
class ChainInfo:
    chain_id: str
    residues_ordered: List[ResidueKey]
    residue_count: int
    min_resseq: int
    max_resseq: int


@dataclass
class JobSpec:
    input_pdb: Path
    pdb_stem: str
    target_chain: str
    binder_chain: str
    binder_len: int
    target_start: int
    target_end: int
    contig: str
    hotspot_arg: Optional[str]
    docker_cmd: List[str]
    out_subdir: Path


@dataclass
class HotspotToken:
    chain: Optional[str]
    resseq: int
    icode: str


def normalize_icode(ch: str) -> str:
    if not ch or ch in {" ", "\0"}:
        return ""
    return str(ch)


def is_protein_residue(res: gemmi.Residue) -> bool:
    if str(res.het_flag) == "H":
        return False

    name = (res.name or "").strip().upper()
    if not name:
        return False

    if name in CANONICAL_AA3:
        return True

    try:
        info = gemmi.find_tabulated_residue(name)
        if info and info.found() and info.is_amino_acid():
            return True
    except Exception:
        pass

    return False


def load_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    if len(st) == 0:
        raise ValueError(f"No model found in structure: {path}")
    try:
        st.remove_alternative_conformations()
    except Exception:
        pass
    return st


def extract_protein_chain_infos(path: Path) -> Dict[str, ChainInfo]:
    st = load_structure(path)
    model = st[0]

    per_chain: Dict[str, List[ResidueKey]] = {}
    seen: Dict[str, set] = {}

    for chain in model:
        chain_id = chain.name if chain.name else "_"
        per_chain.setdefault(chain_id, [])
        seen.setdefault(chain_id, set())
        for res in chain:
            if not is_protein_residue(res):
                continue
            key = ResidueKey(
                chain=chain_id,
                resseq=int(res.seqid.num),
                icode=normalize_icode(res.seqid.icode),
            )
            uniq = (key.resseq, key.icode)
            if uniq in seen[chain_id]:
                continue
            seen[chain_id].add(uniq)
            per_chain[chain_id].append(key)

    chain_infos: Dict[str, ChainInfo] = {}
    for chain_id, residues in per_chain.items():
        if not residues:
            continue
        resseqs = [r.resseq for r in residues]
        chain_infos[chain_id] = ChainInfo(
            chain_id=chain_id,
            residues_ordered=residues,
            residue_count=len(residues),
            min_resseq=min(resseqs),
            max_resseq=max(resseqs),
        )
    return chain_infos


def infer_binder_chain(
    chain_infos: Dict[str, ChainInfo],
    target_chain: str,
    binder_chain: Optional[str],
    pdb_path: Path,
) -> str:
    if target_chain not in chain_infos:
        chains = ", ".join(sorted(chain_infos.keys())) or "(none)"
        raise ValueError(
            f"[{pdb_path.name}] target chain '{target_chain}' not found among protein chains: {chains}"
        )

    if binder_chain:
        if binder_chain == target_chain:
            raise ValueError(
                f"[{pdb_path.name}] --binder-chain cannot be the same as --target-chain ({target_chain})."
            )
        if binder_chain not in chain_infos:
            chains = ", ".join(sorted(chain_infos.keys())) or "(none)"
            raise ValueError(
                f"[{pdb_path.name}] binder chain '{binder_chain}' not found among protein chains: {chains}"
            )
        return binder_chain

    candidates = [ci for ch, ci in chain_infos.items() if ch != target_chain]
    if not candidates:
        raise ValueError(
            f"[{pdb_path.name}] no non-target protein chain found; please specify --binder-chain explicitly."
        )

    candidates.sort(key=lambda x: x.residue_count, reverse=True)
    best = candidates[0]
    ties = [c for c in candidates if c.residue_count == best.residue_count]
    if len(ties) > 1:
        tie_names = ", ".join(sorted(c.chain_id for c in ties))
        raise ValueError(
            f"[{pdb_path.name}] multiple binder candidates with equal residue_count={best.residue_count}: "
            f"{tie_names}. Please set --binder-chain."
        )
    return best.chain_id


def renumber_structure_for_partial_diffusion(
    input_pdb: Path,
    output_pdb: Path,
    target_chain: str,
    renumber_all: bool,
) -> Dict[ResidueKey, ResidueKey]:
    st = load_structure(input_pdb)
    model = st[0]
    mapping: Dict[ResidueKey, ResidueKey] = {}

    for chain in model:
        chain_id = chain.name if chain.name else "_"
        if renumber_all:
            should_renumber_chain = True
        else:
            should_renumber_chain = chain_id == target_chain

        if not should_renumber_chain:
            continue

        new_idx = 1
        for res in chain:
            if not is_protein_residue(res):
                continue
            old_key = ResidueKey(
                chain=chain_id,
                resseq=int(res.seqid.num),
                icode=normalize_icode(res.seqid.icode),
            )
            res.seqid = gemmi.SeqId(new_idx, " ")
            new_key = ResidueKey(chain=chain_id, resseq=new_idx, icode="")
            mapping[old_key] = new_key
            new_idx += 1

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(output_pdb))
    return mapping


def parse_hotspots(raw: str) -> List[HotspotToken]:
    toks: List[HotspotToken] = []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("--hotspots is empty after parsing.")

    for tok in items:
        m_num = NUMERIC_HOTSPOT_RE.fullmatch(tok)
        if m_num:
            toks.append(
                HotspotToken(
                    chain=None,
                    resseq=int(m_num.group(1)),
                    icode=normalize_icode(m_num.group(2)),
                )
            )
            continue

        m_ch = CHAINED_HOTSPOT_RE.fullmatch(tok)
        if m_ch:
            toks.append(
                HotspotToken(
                    chain=m_ch.group(1),
                    resseq=int(m_ch.group(2)),
                    icode=normalize_icode(m_ch.group(3)),
                )
            )
            continue

        raise ValueError(
            f"Invalid hotspot token '{tok}'. Supported forms: '94', '94A', 'B94', 'B94A'."
        )
    return toks


def map_hotspot_to_renumbered(
    old_chain: str,
    old_resseq: int,
    old_icode: str,
    mapping: Dict[ResidueKey, ResidueKey],
) -> ResidueKey:
    direct = ResidueKey(old_chain, old_resseq, old_icode)
    if direct in mapping:
        return mapping[direct]

    if old_icode:
        raise ValueError(
            f"Hotspot {old_chain}{old_resseq}{old_icode} not found in renumber mapping."
        )

    cands = [v for k, v in mapping.items() if k.chain == old_chain and k.resseq == old_resseq]
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise ValueError(
            f"Hotspot {old_chain}{old_resseq} is ambiguous due to insertion codes; please specify explicit insertion code."
        )
    raise ValueError(f"Hotspot {old_chain}{old_resseq} not found in renumber mapping.")


def build_hotspot_arg(
    raw_hotspots: Optional[str],
    target_chain: str,
    remap_hotspot_chain: bool,
    renumber_mapping: Optional[Dict[ResidueKey, ResidueKey]],
) -> Optional[str]:
    if not raw_hotspots:
        return None

    tokens = parse_hotspots(raw_hotspots)
    out_tokens: List[str] = []
    renumber_mapping = renumber_mapping or {}

    for tok in tokens:
        chain = tok.chain if tok.chain is not None else target_chain
        if remap_hotspot_chain:
            chain = target_chain

        resseq = tok.resseq
        icode = tok.icode

        key = ResidueKey(chain=chain, resseq=resseq, icode=icode)
        if key in renumber_mapping or (not icode and any(k.chain == chain and k.resseq == resseq for k in renumber_mapping)):
            mapped = map_hotspot_to_renumbered(chain, resseq, icode, renumber_mapping)
            chain, resseq, icode = mapped.chain, mapped.resseq, mapped.icode

        out_tokens.append(f"{chain}{resseq}{icode}")

    return "ppi.hotspot_res=[" + ",".join(out_tokens) + "]"


def build_contig(binder_len: int, target_chain: str, target_start: int, target_end: int) -> str:
    return f"contigmap.contigs=[{binder_len}-{binder_len}/0 {target_chain}{target_start}-{target_end}]"


def format_cmd(cmd: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(cmd))


def setup_logger(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "run.log"
    logger = logging.getLogger("run_partialdiffusion_inbatch")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def append_summary_row(
    summary_path: Path,
    row: Dict[str, str],
    lock: threading.Lock,
) -> None:
    header = [
        "timestamp",
        "pdb",
        "status",
        "message",
        "target_chain",
        "binder_chain",
        "binder_len",
        "target_start",
        "target_end",
        "renumber_mode",
        "contig",
        "hotspots",
        "job_outdir",
        "input_for_inference",
        "docker_cmd",
    ]

    with lock:
        need_header = (not summary_path.exists()) or summary_path.stat().st_size == 0
        with summary_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter="\t")
            if need_header:
                writer.writeheader()
            writer.writerow(row)


def ensure_target_span_contiguous(target_info: ChainInfo, pdb_name: str) -> None:
    span = target_info.max_resseq - target_info.min_resseq + 1
    if span != target_info.residue_count:
        raise ValueError(
            f"[{pdb_name}] target chain {target_info.chain_id} numbering is not contiguous: "
            f"min={target_info.min_resseq}, max={target_info.max_resseq}, span={span}, actual={target_info.residue_count}. "
            f"Use --renumber-target-to-1 or --renumber-all-to-1."
        )


def container_input_path(
    inference_input: Path,
    input_dir: Path,
    output_dir: Path,
) -> str:
    in_abs = inference_input.resolve()
    in_root = input_dir.resolve()
    out_root = output_dir.resolve()
    try:
        rel = in_abs.relative_to(in_root)
        return f"/inputs/{rel.as_posix()}"
    except ValueError:
        pass
    try:
        rel = in_abs.relative_to(out_root)
        return f"/outputs/{rel.as_posix()}"
    except ValueError:
        pass
    raise ValueError(
        f"inference input path must be under input_dir or output_dir, got: {inference_input}"
    )


def build_docker_cmd(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    input_for_inference: Path,
    output_prefix_container: str,
    contig: str,
    hotspot_arg: Optional[str],
) -> List[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
        "-v",
        f"{input_dir.resolve()}:/inputs",
        "-v",
        f"{output_dir.resolve()}:/outputs",
        args.docker_image,
        f"inference.input_pdb={container_input_path(input_for_inference, input_dir, output_dir)}",
        f"inference.output_prefix={output_prefix_container}",
        f"inference.num_designs={int(args.num_designs)}",
        contig,
        f"diffuser.partial_T={int(args.partial_T)}",
        f"denoiser.noise_scale_ca={float(args.noise_scale_ca)}",
        f"denoiser.noise_scale_frame={float(args.noise_scale_frame)}",
    ]
    if hotspot_arg:
        cmd.append(hotspot_arg)
    return cmd


def preflight_check(args: argparse.Namespace, logger: logging.Logger) -> None:
    logger.info("Preflight: checking docker executable...")
    subprocess.run(["docker", "--version"], check=True)

    logger.info("Preflight: checking GPU visibility inside docker image...")
    probe_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}",
        args.docker_image,
        "python",
        "-c",
        "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)",
    ]
    subprocess.run(probe_cmd, check=True)
    logger.info("Preflight passed.")


def collect_input_pdbs(input_dir: Path) -> List[Path]:
    pdbs = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdb"]
    pdbs.sort(key=lambda x: x.name.lower())
    return pdbs


def prepare_job(
    pdb_path: Path,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
) -> Tuple[JobSpec, str, Path]:
    original_infos = extract_protein_chain_infos(pdb_path)
    target_chain = args.target_chain
    binder_chain = infer_binder_chain(
        chain_infos=original_infos,
        target_chain=target_chain,
        binder_chain=args.binder_chain,
        pdb_path=pdb_path,
    )

    binder_len = original_infos[binder_chain].residue_count
    renumber_mode = "none"
    renumber_mapping: Dict[ResidueKey, ResidueKey] = {}
    inference_input = pdb_path

    if args.renumber_all_to_1 or args.renumber_target_to_1:
        renumber_mode = "all" if args.renumber_all_to_1 else "target_only"
        tmp_input = output_dir / "_tmp_inputs" / f"{pdb_path.stem}.pdb"
        renumber_mapping = renumber_structure_for_partial_diffusion(
            input_pdb=pdb_path,
            output_pdb=tmp_input,
            target_chain=target_chain,
            renumber_all=bool(args.renumber_all_to_1),
        )
        inference_input = tmp_input

    effective_infos = extract_protein_chain_infos(inference_input)
    if target_chain not in effective_infos:
        raise ValueError(f"[{pdb_path.name}] target chain '{target_chain}' missing after preprocessing.")
    if binder_chain not in effective_infos:
        raise ValueError(f"[{pdb_path.name}] binder chain '{binder_chain}' missing after preprocessing.")

    target_info = effective_infos[target_chain]
    ensure_target_span_contiguous(target_info, pdb_path.name)

    target_start = target_info.min_resseq
    target_end = target_info.max_resseq

    contig = build_contig(
        binder_len=binder_len,
        target_chain=target_chain,
        target_start=target_start,
        target_end=target_end,
    )
    hotspot_arg = build_hotspot_arg(
        raw_hotspots=args.hotspots,
        target_chain=target_chain,
        remap_hotspot_chain=bool(args.remap_hotspot_chain),
        renumber_mapping=renumber_mapping,
    )

    out_subdir = output_dir / pdb_path.stem
    output_prefix_container = f"/outputs/{pdb_path.stem}/{pdb_path.stem}"
    docker_cmd = build_docker_cmd(
        args=args,
        input_dir=input_dir,
        output_dir=output_dir,
        input_for_inference=inference_input,
        output_prefix_container=output_prefix_container,
        contig=contig,
        hotspot_arg=hotspot_arg,
    )

    spec = JobSpec(
        input_pdb=pdb_path,
        pdb_stem=pdb_path.stem,
        target_chain=target_chain,
        binder_chain=binder_chain,
        binder_len=binder_len,
        target_start=target_start,
        target_end=target_end,
        contig=contig,
        hotspot_arg=hotspot_arg,
        docker_cmd=docker_cmd,
        out_subdir=out_subdir,
    )
    return spec, renumber_mode, inference_input


def flatten_pdb_outputs(job_outdir: Path, flat_dir: Path, pdb_stem: str) -> int:
    flat_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(job_outdir.rglob("*.pdb")):
        rel_parts = src.relative_to(job_outdir).parts
        if any(part.lower() == "traj" for part in rel_parts[:-1]):
            continue
        if src.name.lower().endswith("_traj.pdb"):
            continue

        dst_name = f"{pdb_stem}__{src.name}"
        dst = flat_dir / dst_name
        if dst.exists():
            k = 2
            while True:
                alt = flat_dir / f"{pdb_stem}__{k}__{src.name}"
                if not alt.exists():
                    dst = alt
                    break
                k += 1
        shutil.copy2(src, dst)
        copied += 1
    return copied


def run_one_job(
    spec: JobSpec,
    args: argparse.Namespace,
    logger: logging.Logger,
    summary_path: Path,
    summary_lock: threading.Lock,
    renumber_mode: str,
    inference_input: Path,
    output_root: Path,
) -> None:
    spec.out_subdir.mkdir(parents=True, exist_ok=True)

    cmd_text = format_cmd(spec.docker_cmd)
    logger.info(
        "[%s] prepared | target=%s binder=%s binder_len=%d target_range=%s%d-%d",
        spec.pdb_stem,
        spec.target_chain,
        spec.binder_chain,
        spec.binder_len,
        spec.target_chain,
        spec.target_start,
        spec.target_end,
    )
    logger.info("[%s] contig: %s", spec.pdb_stem, spec.contig)
    if spec.hotspot_arg:
        logger.info("[%s] hotspots: %s", spec.pdb_stem, spec.hotspot_arg)
    logger.info("[%s] docker: %s", spec.pdb_stem, cmd_text)

    ts = datetime.now().isoformat(timespec="seconds")
    status = "DRY_RUN" if args.dry_run else "OK"
    msg = ""

    try:
        if not args.dry_run:
            subprocess.run(spec.docker_cmd, check=True)

            if args.flatten:
                copied = flatten_pdb_outputs(
                    job_outdir=spec.out_subdir,
                    flat_dir=output_root / "ALL_PDB",
                    pdb_stem=spec.pdb_stem,
                )
                logger.info("[%s] flatten copied %d pdb files.", spec.pdb_stem, copied)
                msg = f"flatten_copied={copied}"
        else:
            logger.info("[%s] dry-run only, command not executed.", spec.pdb_stem)
    except Exception as e:
        status = "FAILED"
        msg = str(e)
        logger.error("[%s] FAILED: %s", spec.pdb_stem, e)
        raise
    finally:
        append_summary_row(
            summary_path=summary_path,
            lock=summary_lock,
            row={
                "timestamp": ts,
                "pdb": spec.input_pdb.name,
                "status": status,
                "message": msg,
                "target_chain": spec.target_chain,
                "binder_chain": spec.binder_chain,
                "binder_len": str(spec.binder_len),
                "target_start": str(spec.target_start),
                "target_end": str(spec.target_end),
                "renumber_mode": renumber_mode,
                "contig": spec.contig,
                "hotspots": spec.hotspot_arg or "",
                "job_outdir": str(spec.out_subdir),
                "input_for_inference": str(inference_input),
                "docker_cmd": cmd_text,
            },
        )


def run_jobs(
    prepared_jobs: List[Tuple[JobSpec, str, Path]],
    args: argparse.Namespace,
    logger: logging.Logger,
    summary_path: Path,
    output_root: Path,
) -> None:
    summary_lock = threading.Lock()

    if int(args.max_workers) <= 1:
        for spec, renumber_mode, inference_input in prepared_jobs:
            run_one_job(
                spec=spec,
                args=args,
                logger=logger,
                summary_path=summary_path,
                summary_lock=summary_lock,
                renumber_mode=renumber_mode,
                inference_input=inference_input,
                output_root=output_root,
            )
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("Running with max_workers=%d", int(args.max_workers))
    first_exc: Optional[BaseException] = None
    futures = {}
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        for spec, renumber_mode, inference_input in prepared_jobs:
            fut = ex.submit(
                run_one_job,
                spec,
                args,
                logger,
                summary_path,
                summary_lock,
                renumber_mode,
                inference_input,
                output_root,
            )
            futures[fut] = spec

        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                fut.result()
            except BaseException as e:
                first_exc = e
                logger.error(
                    "[%s] encountered failure; cancelling not-yet-started tasks (fail-fast).",
                    spec.pdb_stem,
                )
                for other in futures:
                    if other is not fut:
                        other.cancel()
                break

    if first_exc is not None:
        raise RuntimeError(f"Batch failed (fail-fast): {first_exc}") from first_exc


def validate_args(args: argparse.Namespace) -> None:
    if not args.target_chain or not args.target_chain.strip():
        raise ValueError("--target-chain cannot be empty.")
    if args.renumber_target_to_1 and args.renumber_all_to_1:
        raise ValueError("--renumber-target-to-1 and --renumber-all-to-1 are mutually exclusive.")
    if int(args.max_workers) < 1:
        raise ValueError("--max-workers must be >= 1.")
    if int(args.num_designs) < 1:
        raise ValueError("--num-designs must be >= 1.")
    if int(args.partial_T) < 0:
        raise ValueError("--partial-T must be >= 0.")

    if args.binder_chain and args.binder_chain == args.target_chain:
        raise ValueError("--binder-chain cannot equal --target-chain.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch run RFdiffusion partial diffusion on complex PDBs.",
    )
    p.add_argument("--input-dir", required=True, help="Directory containing input .pdb files.")
    p.add_argument("--output-dir", required=True, help="Output root directory.")
    p.add_argument("--target-chain", required=True, help="Target chain ID (e.g., A or B).")
    p.add_argument("--binder-chain", default=None, help="Optional binder chain ID override.")
    p.add_argument(
        "--hotspots",
        default=None,
        help="Global hotspots, e.g. '94,127,139' or 'B94,B127'. Mixed forms supported.",
    )
    p.add_argument(
        "--remap-hotspot-chain",
        action="store_true",
        help="If set, force hotspot chain IDs to target-chain.",
    )
    p.add_argument("--num-designs", type=int, default=10, help="RFdiffusion inference.num_designs.")
    p.add_argument("--partial-T", type=int, default=20, help="RFdiffusion diffuser.partial_T.")
    p.add_argument("--noise-scale-ca", type=float, default=0.5, help="denoiser.noise_scale_ca.")
    p.add_argument("--noise-scale-frame", type=float, default=0.5, help="denoiser.noise_scale_frame.")
    p.add_argument("--cuda-visible-devices", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    p.add_argument(
        "--docker-image",
        default="rosettacommons/rfdiffusion:latest",
        help="RFdiffusion docker image.",
    )
    p.add_argument(
        "--renumber-target-to-1",
        action="store_true",
        help="Renumber only target chain protein residues to 1..N in temp input.",
    )
    p.add_argument(
        "--renumber-all-to-1",
        action="store_true",
        help="Renumber all protein chains to 1..N per chain in temp input.",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Copy generated .pdb files into output_dir/ALL_PDB with unique names.",
    )
    p.add_argument("--max-workers", type=int, default=1, help="Parallel workers. Default: 1.")
    p.add_argument("--dry-run", action="store_true", help="Print and log commands only.")
    p.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="Skip docker/GPU preflight checks.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.target_chain = args.target_chain.strip()
    if args.binder_chain is not None:
        args.binder_chain = args.binder_chain.strip()
    validate_args(args)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir not found or not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    summary_path = output_dir / "run_summary.tsv"

    pdb_files = collect_input_pdbs(input_dir)
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in --input-dir: {input_dir}")

    logger.info("Found %d input PDB files in %s", len(pdb_files), input_dir)
    logger.info("Output root: %s", output_dir)

    if (not args.dry_run) and (not args.skip_gpu_preflight):
        preflight_check(args, logger)
    else:
        if args.dry_run:
            logger.info("Dry-run mode: skipping GPU preflight.")
        if args.skip_gpu_preflight:
            logger.info("--skip-gpu-preflight set: preflight skipped.")

    prepared_jobs: List[Tuple[JobSpec, str, Path]] = []
    for pdb_path in pdb_files:
        prepared = prepare_job(
            pdb_path=pdb_path,
            args=args,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        prepared_jobs.append(prepared)

    logger.info("Prepared %d jobs. Starting execution...", len(prepared_jobs))
    run_jobs(
        prepared_jobs=prepared_jobs,
        args=args,
        logger=logger,
        summary_path=summary_path,
        output_root=output_dir,
    )
    logger.info("All jobs completed successfully.")


if __name__ == "__main__":
    main()
