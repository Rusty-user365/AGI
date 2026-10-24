"""Experiment 6 — the structural guard (C1), tested without the editor.

C1 is a claim about the update pipeline, not about generation:

    a verified-wrong example cannot move the plastic substrate and cannot
    create a factual memory entry.

C2 (does the guard improve held-out behaviour?) is a different claim, measured
by the NLL diff-in-diff in exp2, and it is unresolved because the editor is at
the floor. Conflating the two is what made the earlier assessment too
pessimistic: C1 is testable today and it is the claim the paper rests on.

Four checks, pre-registered in PAPER_REVISION.md §5.5 *before* this ran:

  A  gradient masking        max |grad| on plastic_a/plastic_b is exactly 0.0
  B  parameter immutability  torch.equal on both plastic tensors
  C  memory write guard      no entry created for the verified-wrong claim
  D  rejection learning      gate gradient > 1e-8, so the guard does not also
                             switch off the learning it should permit

Two controls validate the instrument:

  unguarded      the same wrong claims treated as verified. A, B and C must
                 FAIL here, or the test cannot detect the leak it claims to.
  verified-right the same pipeline on true claims. Gradient must flow,
                 parameters must change, the entry must be created.

D is the check that catches the tempting bug: if the implementation masked
*everything* for verified-wrong, A would pass while the model learned nothing,
including not to reject. A test that cannot distinguish those is worthless.

One diagnostic is reported but was **not** pre-registered, and is labelled as
such: weight decay under AdamW applies to any parameter whose gradient is not
None, so masking is sufficient only because the masked path leaves the plastic
gradient as None. The diagnostic measures both the None and the zero-tensor
case, because a future loss path that touches the substrate with a zero
contribution would reintroduce drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from _common import arg_parser, banner, get_system, table, write_result
from cgr.data.facts import make_claim_set
from cgr.model.interpreter import adjudicate
from cgr.scaffold import episodes_from_claims
from cgr.train import collate

TOL = 1e-8
MEMORY_CLAIM = "2 + 2 = 5 ."     # adjudicated false under the default scope

PREDICTION = {
    "A": "max |grad| on plastic_a/plastic_b == 0.0 exactly for a "
         "verified-wrong-only batch; a value in (0, 1e-8) is ambiguous",
    "B": "plastic tensors bit-identical (torch.equal) after the update",
    "C": "update_memory(verified=False) returns -1 and creates no entry",
    "D": "gate gradient > 1e-8: the rejection path is supervised, so the "
         "safeguard does not also disable the learning it should allow",
    "unguarded_control": "A, B and C must FAIL on the same wrong claims "
                         "treated as verified",
    "verified_right_control": "gradient flows, parameters change, entry created",
}


def plastic_grads(system) -> torch.Tensor:
    parts = []
    for p in (system.backbone.plastic_a, system.backbone.plastic_b):
        parts.append((p.grad if p.grad is not None
                      else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts)


def gate_grad_norm(system) -> float:
    sq = 0.0
    for p in system.gate.parameters():
        if p.grad is not None:
            sq += float(p.grad.pow(2).sum())
    return sq ** 0.5


def run_update(system, batch, verified, *, lr: float = 0.1,
               seed: int = 0, optim_cls=torch.optim.SGD,
               weight_decay: float = 0.0) -> dict:
    """One consolidation step with instrumentation around it.

    `verified` is either a bool (applied to the whole batch) or one flag per
    example. The per-example form is what the mixed-batch diagnostic needs: a
    mixed batch must be *built* the way the pipeline would build it, with the
    verifier's verdicts, not with everything treated as verified.
    """
    n = int(batch["ids"].shape[0])
    flags = [verified] * n if isinstance(verified, bool) else list(verified)
    if len(flags) != n:
        raise ValueError(f"got {len(flags)} verdicts for {n} examples")
    system.zero_grad(set_to_none=True)
    torch.manual_seed(seed)          # identical denoising mask across conditions
    snap = system.plastic_snapshot()
    opt = optim_cls(system.parameters(), lr=lr, weight_decay=weight_decay)
    st = system.consolidate(batch["ids"], batch["clock"], batch["labels"],
                            verified=flags,
                            advantage=[1.0] * n, optim=opt)
    g = plastic_grads(system)
    return {
        "n": n,
        "n_kept": int(st.get("n_updated", 0)),
        "plastic_grad_max": float(g.abs().max()) if g.numel() else 0.0,
        "plastic_grad_norm": float(g.norm()),
        "plastic_unchanged": system.plastic_unchanged(snap),
        "gate_grad_norm": gate_grad_norm(system),
        "policy_supervised": bool(st.get("policy_supervised", False)),
        "stepped": bool(st.get("stepped", False)),
        "stats": {k: v for k, v in st.items() if isinstance(v, (int, float, bool))},
    }


def memory_check(system, *, verified: bool, allow_unverified: bool = False
                 ) -> dict:
    system.memory.clear()
    before = len(system.memory)
    eid = system.update_memory(MEMORY_CLAIM, verified=verified,
                               allow_unverified=allow_unverified)
    after = len(system.memory)
    return {"returned": int(eid), "before": before, "after": after,
            "created": after > before}


def classify(check: str, value: dict) -> str:
    """Three-valued: pass | fail | ambiguous, per the pre-registration."""
    if check == "A":
        if value["plastic_grad_max"] == 0.0:
            return "pass"
        if value["plastic_grad_max"] < TOL:
            return "ambiguous"
        return "fail"
    if check == "B":
        return "pass" if value["plastic_unchanged"] else "fail"
    if check == "C":
        return "pass" if (value["returned"] == -1 and not value["created"]) \
            else "fail"
    if check == "D":
        if value["gate_grad_norm"] > TOL and value["policy_supervised"]:
            return "pass"
        return "fail"
    raise ValueError(check)


def main() -> int:
    ap = arg_parser("Experiment 6: the structural guard (C1)")
    args = ap.parse_args()
    n = args.n or 16

    banner("Experiment 6 — structural guard (C1), independent of the editor")
    system, cfg = get_system(args.preset, seed=args.seed,
                             checkpoint=args.checkpoint)
    system.eval()

    # All parameters trainable: the test must be able to detect a leak, and a
    # frozen parameter cannot leak. exp1 freezes everything but the adapter,
    # which is why it can only check that nothing moved, not that something
    # else could have moved.
    for p in system.parameters():
        p.requires_grad_(True)

    pool = make_claim_set(n * 3, seed=args.seed + 7, marker_present=True)
    wrong = [c for c in pool
             if adjudicate(c.spec, c.marker_present).correct is False][:n]
    right = [c for c in pool
             if adjudicate(c.spec, c.marker_present).correct is True][:n]
    if len(wrong) < n or len(right) < n:
        raise SystemExit(f"not enough adjudicable claims ({len(wrong)} wrong, "
                         f"{len(right)} right); raise n")

    wrong_batch = collate(episodes_from_claims(wrong, system.tok),
                          system.tok, cfg.model.max_len)
    right_batch = collate(episodes_from_claims(right, system.tok),
                          system.tok, cfg.model.max_len)

    # ---------------------------------------------------------------- checks
    guarded = run_update(system, wrong_batch, verified=False, seed=args.seed)
    verdicts = {"A": classify("A", guarded), "B": classify("B", guarded),
                "D": classify("D", guarded)}
    mem_guarded = memory_check(system, verified=False)
    verdicts["C"] = classify("C", mem_guarded)

    # ------------------------------------------------------------- controls
    # Unguarded: the same wrong claims, no verifier. Every check must fail.
    unguarded = run_update(system, wrong_batch, verified=True, seed=args.seed)
    mem_unguarded = memory_check(system, verified=True)
    control_unguarded = {
        "A_fails": unguarded["plastic_grad_max"] > TOL,
        "B_fails": not unguarded["plastic_unchanged"],
        "C_fails": mem_unguarded["created"],
        "grad_max": unguarded["plastic_grad_max"],
        "drifted": not unguarded["plastic_unchanged"],
        "entry_created": mem_unguarded["created"],
    }

    # Verified-right: the pipeline is not simply broken.
    right_run = run_update(system, right_batch, verified=True, seed=args.seed)
    mem_right = memory_check(system, verified=True)
    control_right = {
        "gradient_flows": right_run["plastic_grad_max"] > TOL,
        "parameters_change": not right_run["plastic_unchanged"],
        "entry_created": mem_right["created"],
        "grad_max": right_run["plastic_grad_max"],
    }

    # ------------------------------------- extra (NOT pre-registered) checks
    # A2: a mixed batch, assembled the way the pipeline would assemble it,
    # must give the substrate exactly the gradient it would get from the
    # verified-right examples alone. Three confounds have to be removed or the
    # check measures something else:
    #   - the wrong claims must be marked *unverified* (the first version
    #     marked them verified, which is the unguarded condition);
    #   - right examples go first, so the kept subset is row-aligned;
    #   - `run_update` steps, so the two gradients must be measured from the
    #     same parameter state (snapshot/restore), and both batches must have
    #     the same padded width, because `collate` pads to the batch max and
    #     the denoising mask is drawn on the padded tensor.
    state = {k: v.detach().clone() for k, v in system.state_dict().items()}

    def same_width(batch: dict, width: int) -> dict:
        ids = batch["ids"]
        if ids.shape[1] < width:
            pad = torch.full((ids.shape[0], width - ids.shape[1]), system.tok.pad_id,
                             dtype=torch.long)
            batch = dict(batch, ids=torch.cat([ids, pad], dim=1))
        return batch

    mixed_batch = collate(episodes_from_claims(right + wrong, system.tok),
                          system.tok, cfg.model.max_len)
    width = mixed_batch["ids"].shape[1]
    right_same = same_width(collate(episodes_from_claims(right, system.tok),
                                    system.tok, cfg.model.max_len), width)
    mixed_flags = [True] * len(right) + [False] * len(wrong)

    mixed_run = run_update(system, mixed_batch, mixed_flags, seed=args.seed)
    mixed_grad = plastic_grads(system).clone()
    system.load_state_dict(state)

    right_run2 = run_update(system, right_same, verified=True, seed=args.seed)
    right_grad = plastic_grads(system).clone()
    system.load_state_dict(state)

    a2 = float((mixed_grad - right_grad).abs().max())
    a2_row_aligned = mixed_run["n_kept"] == len(right)
    a2_width_matched = right_same["ids"].shape[1] == mixed_batch["ids"].shape[1]

    # B2: weight decay under AdamW. Two distinct cases, because PyTorch's
    # AdamW skips parameters whose grad is None but decays parameters whose
    # grad is a zero tensor. "Masked" produces the former; a loss path that
    # touches the substrate with a zero contribution can produce the latter.
    # Tested on standalone tensors so the checkpoint is not perturbed.
    def wd_moves(grad_kind: str) -> bool:
        p = torch.nn.Parameter(torch.randn(4, 4))
        before = p.detach().clone()
        opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.01)
        p.grad = (torch.zeros_like(p) if grad_kind == "zeros" else None)
        opt.step()
        return not torch.equal(before, p.detach())

    b2 = {"grad_none_moves": wd_moves("none"),
          "grad_zeros_moves": wd_moves("zeros")}
    extra = {
        "note": "not pre-registered; reported as diagnostics, not pass/fail",
        "A2_mixed_vs_right_only_max_grad_diff": a2,
        "A2_holds": a2 == 0.0,
        "A2_row_aligned": a2_row_aligned,
        "A2_width_matched": a2_width_matched,
        "A2_mixed_n": mixed_run["n"], "A2_kept": mixed_run["n_kept"],
        "B2_adamw_wd": b2,
        "B2_note": "PyTorch AdamW decays only parameters with a non-None "
                   "grad. Masking leaves plastic grads as None, so decay "
                   "cannot move the substrate. If any code path touches the "
                   "substrate and yields a zero tensor instead, decay does "
                   "move it; a production optimizer should exclude the "
                   "plastic parameters from weight decay.",
    }

    # ------------------------------------------------------------ reporting
    rows = [
        {"check": "A gradient masking", "value": guarded["plastic_grad_max"],
         "verdict": verdicts["A"]},
        {"check": "B immutability", "value": float(guarded["plastic_unchanged"]),
         "verdict": verdicts["B"]},
        {"check": "C memory guard", "value": float(not mem_guarded["created"]),
         "verdict": verdicts["C"]},
        {"check": "D rejection learning", "value": guarded["gate_grad_norm"],
         "verdict": verdicts["D"]},
    ]
    table(rows, ["check", "value", "verdict"])

    ctrl_rows = [
        {"control": "unguarded", "A_fails": control_unguarded["A_fails"],
         "B_fails": control_unguarded["B_fails"],
         "C_fails": control_unguarded["C_fails"]},
        {"control": "verified-right", "A_fails": "-",
         "B_fails": "-", "C_fails": "-"},
    ]
    table(ctrl_rows, ["control", "A_fails", "B_fails", "C_fails"])

    instrument_valid = (control_unguarded["A_fails"]
                        and control_unguarded["B_fails"]
                        and control_unguarded["C_fails"])
    c1_pass = all(v == "pass" for v in verdicts.values())
    c1_ambiguous = any(v == "ambiguous" for v in verdicts.values())

    notes = []
    if not instrument_valid:
        notes.append(
            "INSTRUMENT INVALID: the unguarded control did not show the leak. "
            "A, B and C passing means nothing, because the gradient was not "
            "reaching the parameters to begin with. Fix the test before "
            "reporting C1 either way.")
    if c1_pass and instrument_valid:
        notes.append(
            "C1 passes with a valid instrument: verified-wrong examples leave "
            "the plastic substrate bit-identical and create no memory entry, "
            "while the gate still receives a rejection gradient. This is a "
            "structural guarantee about the pipeline and is independent of "
            "whether the editor can generate good repairs.")
    elif c1_ambiguous:
        notes.append(
            "C1 is AMBIGUOUS: a plastic gradient was non-zero but below the "
            "pre-registered tolerance. Reported as ambiguous rather than "
            "rounded to pass.")
    elif not c1_pass:
        failed = [k for k, v in verdicts.items() if v == "fail"]
        notes.append(
            f"C1 FAILS on {failed}. Do not report selective plasticity as a "
            f"guarantee until the leak is fixed.")
    if not control_right["gradient_flows"]:
        notes.append(
            "The verified-right control shows no gradient on the substrate: "
            "the pipeline cannot update at all, so the masking result is "
            "uninterpretable.")
    if not extra["A2_holds"]:
        notes.append(
            f"A2 (not pre-registered): a mixed batch changed the substrate "
            f"gradient by {a2:.3e} relative to the verified-right examples "
            f"alone, so verified-wrong examples are not perfectly inert in a "
            f"mixed batch (row-aligned: {a2_row_aligned}).")
    if extra["B2_adamw_wd"]["grad_zeros_moves"]:
        notes.append(
            "B2 (not pre-registered): under AdamW with weight decay, a "
            "*zero-tensor* gradient still moves the parameter, because decay "
            "is applied whenever grad is not None. Masking leaves the plastic "
            "grad as None, which AdamW skips, so the guard holds on this path "
            "— but a production optimizer should exclude the plastic "
            "parameters from weight decay rather than rely on that.")
    notes.append(
        "C2 (does the guard improve held-out behaviour?) is a different claim "
        "and is unresolved: exp2's NLL diff-in-diff is within its minimum "
        "detectable effect at this scale. C1 does not depend on it.")

    write_result("exp6_structural_guard", {
        "experiment": "exp6_structural_guard",
        "preset": args.preset, "seed": args.seed, "n": n,
        "preregistered_prediction": PREDICTION,
        "preregistration": "PAPER_REVISION.md §5.5, written before the run",
        "tolerance": TOL,
        "checks": {"A": guarded, "C_memory": mem_guarded},
        "verdicts": verdicts,
        "controls": {"unguarded": control_unguarded,
                     "verified_right": control_right},
        "extra_checks": extra,
        "summary": {
            "c1_pass": bool(c1_pass),
            "c1_ambiguous": bool(c1_ambiguous),
            "instrument_valid": bool(instrument_valid),
            "memory_guard_enforced": bool(verdicts["C"] == "pass"),
            "rejection_learning_supervised": bool(verdicts["D"] == "pass"),
        },
        "notes": notes,
    })
    print("\n" + "\n".join(f"  ! {x}" for x in notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
